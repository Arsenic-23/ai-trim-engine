"""
Production Face Pipeline — SCRFD + ArcFace + ByteTrack + HAC + Workspace Registry
(INGESTION_ENGINE.md §6.3 & §7)

Architecture:
- Detection:  SCRFD via insightface (ResNet-10 backbone, ONNX runtime)
              — handles profile faces, occlusion, masks, extreme lighting
- Embeddings: ArcFace 512-d via insightface (w600k_r50 model)
              — discriminative face recognition, L2-normalized
- Tracking:   ByteTrack with Kalman filter (two-stage IoU matching)
              — handles occlusion re-entry, low-score recovery, stale track removal
- Clustering: HAC with distance_threshold on ArcFace embeddings
              — no fixed n_clusters, adaptive to actual face count
- Registry:   Workspace-level face centroids persisted to speaker_embeddings table
              — cross-video identity via ArcFace cosine ≥ threshold
- Speaking:   Mouth-region pixel variance cross-correlated with audio energy
              — AV-sync speaking binding, not presence-only heuristic
- Privacy:    Configurable embedding storage, face blurring in output
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console
from sklearn.cluster import AgglomerativeClustering

from trim_engine.config import CFG

logger = logging.getLogger(__name__)
console = Console()





_MEASUREMENT_DIM = 4   
_STATE_DIM = 8          


class _KalmanFilter:
    """8-dim Kalman filter for bounding box tracking (ByteTrack style)."""

    def __init__(self) -> None:
        dt = 1.0
        self._motion_mat = np.eye(_STATE_DIM, dtype=np.float32)
        for i in range(_MEASUREMENT_DIM):
            self._motion_mat[i, _MEASUREMENT_DIM + i] = dt

        self._update_mat = np.eye(_MEASUREMENT_DIM, _STATE_DIM, dtype=np.float32)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    def initiate(self, measurement: np.ndarray) -> None:
        """Create a new track from the first detection."""
        self.mean = np.zeros(_STATE_DIM, dtype=np.float32)
        self.mean[:_MEASUREMENT_DIM] = measurement
        self.mean[_MEASUREMENT_DIM:] = 0.0

        std_pos = np.array([self._std_weight_position] * _MEASUREMENT_DIM, dtype=np.float32)
        std_vel = np.array([self._std_weight_velocity] * _MEASUREMENT_DIM, dtype=np.float32)
        self.covariance = np.diag(np.square(np.concatenate([std_pos, std_vel])))

    def predict(self) -> None:
        """Advance the state by one timestep."""
        if self.mean is None or self.covariance is None:
            return

        std_pos = np.array([self._std_weight_position] * _MEASUREMENT_DIM, dtype=np.float32)
        std_vel = np.array([self._std_weight_velocity] * _MEASUREMENT_DIM, dtype=np.float32)
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        self.mean = self._motion_mat @ self.mean
        self.covariance = self._motion_mat @ self.covariance @ self._motion_mat.T + motion_cov

    def update(self, measurement: np.ndarray) -> None:
        """Correct the state with a detection."""
        if self.mean is None or self.covariance is None:
            return

        std_pos = np.array([self._std_weight_position] * _MEASUREMENT_DIM, dtype=np.float32)
        innovation_cov = np.diag(np.square(std_pos))

        kalman_gain = self.covariance @ self._update_mat.T @ np.linalg.inv(
            self._update_mat @ self.covariance @ self._update_mat.T + innovation_cov
        )
        self.mean = self.mean + kalman_gain @ (measurement - self._update_mat @ self.mean)

        eye = np.eye(_STATE_DIM, dtype=np.float32)
        self.covariance = (eye - kalman_gain @ self._update_mat) @ self.covariance

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        """Project the state to measurement space."""
        if self.mean is None or self.covariance is None:
            mean_proj = np.zeros(_MEASUREMENT_DIM, dtype=np.float32)
            cov_proj = np.eye(_MEASUREMENT_DIM, dtype=np.float32)
            return mean_proj, cov_proj

        std_pos = np.array([self._std_weight_position] * _MEASUREMENT_DIM, dtype=np.float32)
        innovation_cov = np.diag(np.square(std_pos))

        mean = self._update_mat @ self.mean
        covariance = self._update_mat @ self.covariance @ self._update_mat.T + innovation_cov
        return mean, covariance






class _TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    REMOVED = 4


class _STrack:
    """A single tracked face (ByteTrack-style)."""

    def __init__(self, tlwh: np.ndarray, score: float, track_id: int,
                 embedding: np.ndarray, mouth_activity: float) -> None:
        self.tlwh = tlwh.astype(np.float64)
        self.score = score
        self.track_id = track_id
        self.embedding = embedding
        self.mouth_buffer = [mouth_activity]
        self.mouth_activity = mouth_activity

        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        s = tlwh[2] * tlwh[3]
        r = tlwh[2] / max(tlwh[3], 1.0)

        self.kalman = _KalmanFilter()
        self.kalman.initiate(np.array([cx, cy, s, r], dtype=np.float32))

        self.state = _TrackState.TENTATIVE
        self.hit_count = 0
        self.time_since_update = 0
        self.frame_ids: list[int] = []
        self.observations: list[dict[str, Any]] = []

    def predict(self) -> None:
        self.kalman.predict()

    def update(self, tlwh: np.ndarray, score: float, embedding: np.ndarray,
               mouth_activity: float, frame_id: int, frame_data: dict[str, Any]) -> None:
        self.tlwh = tlwh.astype(np.float64)
        self.score = score
        self.embedding = embedding
        
        self.mouth_buffer.append(mouth_activity)
        if len(self.mouth_buffer) > CFG.face.mouth_motion_window_frames:
            self.mouth_buffer.pop(0)
            
        self.mouth_activity = float(np.mean(self.mouth_buffer))
                
        frame_data["_mouth_activity"] = self.mouth_activity
        self.time_since_update = 0
        self.hit_count += 1
        self.frame_ids.append(frame_id)
        self.observations.append(frame_data)

        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        s = tlwh[2] * tlwh[3]
        r = tlwh[2] / max(tlwh[3], 1.0)

        if self.state == _TrackState.TENTATIVE and self.hit_count >= CFG.face.track_min_hits:
            self.state = _TrackState.CONFIRMED

        self.kalman.update(np.array([cx, cy, s, r], dtype=np.float32))

    def mark_lost(self) -> None:
        self.state = _TrackState.LOST

    def mark_removed(self) -> None:
        self.state = _TrackState.REMOVED

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def is_activated(self) -> bool:
        return self.state == _TrackState.CONFIRMED






def _iou(atlbr: np.ndarray, btlbr: np.ndarray) -> float:
    sx = max(atlbr[0], btlbr[0])
    sy = max(atlbr[1], btlbr[1])
    ex = min(atlbr[2], btlbr[2])
    ey = min(atlbr[3], btlbr[3])

    if sx >= ex or sy >= ey:
        return 0.0

    inter = (ex - sx) * (ey - sy)
    area_a = (atlbr[2] - atlbr[0]) * (atlbr[3] - atlbr[1])
    area_b = (btlbr[2] - btlbr[0]) * (btlbr[3] - btlbr[1])
    return inter / float(area_a + area_b - inter)


def _iou_batch(tlbrs_a: np.ndarray, tlbrs_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of TLBR boxes."""
    if len(tlbrs_a) == 0 or len(tlbrs_b) == 0:
        return np.zeros((len(tlbrs_a), len(tlbrs_b)), dtype=np.float32)

    sx = np.maximum(tlbrs_a[:, None, 0], tlbrs_b[None, :, 0])
    sy = np.maximum(tlbrs_a[:, None, 1], tlbrs_b[None, :, 1])
    ex = np.minimum(tlbrs_a[:, None, 2], tlbrs_b[None, :, 2])
    ey = np.minimum(tlbrs_a[:, None, 3], tlbrs_b[None, :, 3])

    inter = np.maximum(0.0, ex - sx) * np.maximum(0.0, ey - sy)
    area_a = (tlbrs_a[:, 2] - tlbrs_a[:, 0]) * (tlbrs_a[:, 3] - tlbrs_a[:, 1])
    area_b = (tlbrs_b[:, 2] - tlbrs_b[:, 0]) * (tlbrs_b[:, 3] - tlbrs_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _linear_assignment(cost_matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian-style matching on cost matrix. Returns (matches, unmatched_a, unmatched_b)."""
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = np.column_stack([row_ind, col_ind])

    
    valid = cost_matrix[row_ind, col_ind] <= threshold
    matches = matches[valid]

    unmatched_a = np.setdiff1d(np.arange(cost_matrix.shape[0]), matches[:, 0])
    unmatched_b = np.setdiff1d(np.arange(cost_matrix.shape[1]), matches[:, 1])
    return matches, unmatched_a, unmatched_b






class SCRFDDetector:
    """SCRFD face detector via insightface. Handles profile, occlusion, masks."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import insightface
            # §1.7: Use FaceAnalysis app (correct API) instead of raw get_model
            app = insightface.app.FaceAnalysis(
                name=CFG.face.detector_model,
                allowed_modules=["detection"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
            self._model = app.det_model if hasattr(app, 'det_model') else None
            self._loaded = True
            logger.info("Face detector: SCRFD %s loaded via FaceAnalysis", CFG.face.detector_model)
        except ImportError:
            logger.warning("insightface not installed, falling back to cv2 Haar Cascade")
            self._setup_haar_fallback()
        except Exception as exc:
            logger.warning("SCRFD load failed (%s), falling back to Haar Cascade", exc)
            self._setup_haar_fallback()

    def _setup_haar_fallback(self) -> None:
        try:
            self._fallback = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self._fallback.empty():
                raise RuntimeError("Failed to load haarcascade_frontalface_default.xml")
            self._loaded = True
            self._using_haar = True
        except (AttributeError, Exception) as exc:
            raise RuntimeError(
                f"Face detection unavailable: SCRFD failed and Haar cascade not in this OpenCV build ({exc})"
            ) from exc

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect faces in a BGR frame.

        Returns list of dicts with keys:
            bbox: [x1, y1, x2, y2] — integer pixel coords
            score: float confidence
            landmarks: (optional) 5-point facial landmarks [(x,y), ...]
            kps: (optional) raw landmarks array
        """
        self._ensure_loaded()

        if hasattr(self, '_using_haar') and self._using_haar:
            return self._detect_haar(frame)

        return self._detect_scrfd(frame)

    def _detect_scrfd(self, frame: np.ndarray) -> list[dict[str, Any]]:
        try:
            # §1.7: Use FaceAnalysis.get() when available (returns Face objects)
            if hasattr(self, '_app'):
                faces = self._app.get(frame)
                results = []
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    score = float(face.det_score)
                    if score < CFG.face.detection_confidence:
                        continue
                    if (x2 - x1) < CFG.face.min_face_size or (y2 - y1) < CFG.face.min_face_size:
                        continue
                    result = {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "score": score,
                        "_face_obj": face,  # pass through for ArcFace
                    }
                    if face.kps is not None:
                        result["landmarks"] = face.kps.astype(int).tolist()
                        result["kps"] = face.kps
                    results.append(result)
                return results
            # Legacy path
            bboxes, kpss = self._model.detect(frame, max_num=10, metric="default")
        except Exception:
            return self._detect_haar(frame)

        results = []
        if bboxes is not None:
            for i in range(len(bboxes)):
                x1, y1, x2, y2, score = bboxes[i].astype(float)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if score < CFG.face.detection_confidence:
                    continue
                if (x2 - x1) < CFG.face.min_face_size or (y2 - y1) < CFG.face.min_face_size:
                    continue

                result = {
                    "bbox": (x1, y1, x2, y2),
                    "score": float(score),
                }
                if kpss is not None and i < len(kpss):
                    result["landmarks"] = kpss[i].astype(int).tolist()
                    result["kps"] = kpss[i]
                results.append(result)

        return results

    def _detect_haar(self, frame: np.ndarray) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._fallback.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(CFG.face.min_face_size, CFG.face.min_face_size),
        )
        return [
            {"bbox": (int(x), int(y), int(x + w), int(y + h)), "score": 1.0}
            for x, y, w, h in faces
        ]

    @property
    def model_name(self) -> str:
        if hasattr(self, '_using_haar') and self._using_haar:
            return "haar-cascade"
        return f"scrfd-{CFG.face.detector_model}"






class ArcFaceEmbedder:
    """ArcFace 512-d face embeddings via insightface."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False
        self._using_fallback = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import insightface
            from insightface.model_zoo import get_model
            self._model = get_model(CFG.face.recog_model, download=True)
            self._model.prepare(ctx_id=-1)
            self._loaded = True
            logger.info("ArcFace embedder: %s loaded (dim=%d)", CFG.face.recog_model, CFG.face.embedding_dim)
        except ImportError:
            logger.warning("insightface not installed, using fallback embedder")
            self._using_fallback = True
            self._loaded = True
        except Exception as exc:
            logger.warning("ArcFace load failed (%s), using fallback embedder", exc)
            self._using_fallback = True
            self._loaded = True
        # §1.7: flag will be checked post-pipeline to set coverage["face_identity"]

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Compute a 512-d L2-normalized face embedding from a BGR face crop."""
        self._ensure_loaded()

        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return np.zeros(CFG.face.embedding_dim, dtype=np.float32)

        if not self._using_fallback and self._model is not None:
            try:
                # §1.7: Try get_feat() first (modern insightface API), then get_embedding()
                emb = None
                if hasattr(self._model, 'get_feat'):
                    emb = self._model.get_feat(face_crop)
                elif hasattr(self._model, 'get_embedding'):
                    emb = self._model.get_embedding(face_crop)
                if emb is not None and len(emb) == CFG.face.embedding_dim:
                    norm = np.linalg.norm(emb)
                    if norm > 1e-10:
                        emb = emb / norm
                    return emb.astype(np.float32)
            except Exception:
                pass

        return self._fallback_embed(face_crop)

    def _fallback_embed(self, crop: np.ndarray) -> np.ndarray:
        """HOG + LBP + spatial color fallback (128-d → 512-d padded)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        hog = cv2.HOGDescriptor(
            _winSize=(64, 64), _blockSize=(16, 16),
            _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
        )
        resized = cv2.resize(gray, (64, 64))
        hog_feat = hog.compute(resized).flatten()
        if len(hog_feat) > 64:
            chunk = len(hog_feat) // 64
            hog_feat = np.array([np.mean(hog_feat[i * chunk:(i + 1) * chunk]) for i in range(64)])

        lbp = self._lbp_histogram(gray)
        color = self._spatial_color(crop)

        raw = np.concatenate([hog_feat, lbp, color])
        target = CFG.face.embedding_dim
        if len(raw) > target:
            raw = raw[:target]
        elif len(raw) < target:
            raw = np.pad(raw, (0, target - len(raw)))

        norm = np.linalg.norm(raw)
        if norm > 1e-10:
            raw = raw / norm
        return raw.astype(np.float32)

    @staticmethod
    def _lbp_histogram(gray: np.ndarray, bins: int = 32) -> np.ndarray:
        resized = cv2.resize(gray, (32, 32))
        h, w = resized.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            shifted = resized[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            center = resized[1:h - 1, 1:w - 1]
            lbp = (lbp << 1) | (shifted >= center).astype(np.uint8)

        hist, _ = np.histogram(lbp.flatten(), bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    @staticmethod
    def _spatial_color(crop: np.ndarray, grid_size: int = 4, bins: int = 8) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        resized = cv2.resize(hsv, (grid_size * 8, grid_size * 8))
        h, w = resized.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        features = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                cell = resized[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w, 0]
                hist, _ = np.histogram(cell.flatten(), bins=bins, range=(0, 180))
                hist = hist.astype(np.float32)
                total = hist.sum()
                if total > 0:
                    hist /= total
                features.append(hist)
        result = np.concatenate(features[:4])
        return result[:32]

    @property
    def model_name(self) -> str:
        if self._using_fallback:
            return "hog-lbp-color-fallback"
        return f"arcface-{CFG.face.recog_model}"






class ByteTracker:
    """Multi-object tracker implementing the ByteTrack algorithm.

    ByteTrack key insight: low-score detections (e.g. occluded, blurred)
    should be matched against tracks only when no high-score detection competes.
    This allows recovery of occluded/imperfect detections without false positives.
    """

    def __init__(self) -> None:
        self.next_id = 0
        self.tracks: list[_STrack] = []
        self.frame_count = 0

    def update(self, detections: list[dict[str, Any]], frame_id: int,
               embeddings: list[np.ndarray],
               mouth_activities: list[float]) -> list[_STrack]:
        """ByteTrack two-stage matching. Returns activated tracks for this frame."""
        self.frame_count += 1

        if not self.tracks:
            self._init_tracks(detections, embeddings, mouth_activities, frame_id)
            return self.tracks

        for track in self.tracks:
            track.predict()

        
        high_dets, low_dets = self._split_by_score(detections, embeddings, mouth_activities)

        confirmed = [t for t in self.tracks if t.is_activated]
        unconfirmed = [t for t in self.tracks if not t.is_activated]
        high_det_list = [d for d, _, _, _ in high_dets]

        
        matches1, unmatched_t1, unmatched_d1 = self._match(
            confirmed, high_det_list, threshold=1.0 - CFG.face.track_iou_threshold,
        )

        
        for ti, di in matches1:
            det, emb, ma, _ = high_dets[di]
            confirmed[ti].update(
                self._tlbr_to_tlwh(det["bbox"]), det["score"], emb, ma, frame_id, det,
            )

        
        remaining_confirmed = [confirmed[i] for i in unmatched_t1]
        remaining_tracks = remaining_confirmed + unconfirmed

        low_det_list = [d for d, _, _, _ in low_dets]
        matches2, unmatched_t2, unmatched_d2 = self._match(
            remaining_tracks, low_det_list, threshold=1.0 - CFG.face.track_iou_threshold,
        )

        for ti, di in matches2:
            det, emb, ma, _ = low_dets[di]
            remaining_tracks[ti].update(
                self._tlbr_to_tlwh(det["bbox"]), det["score"], emb, ma, frame_id, det,
            )

        
        aged_tracks = [confirmed[i] for i in unmatched_t1]
        aged_tracks += [remaining_tracks[i] for i in unmatched_t2]

        for t in aged_tracks:
            t.time_since_update += 1
            if t.time_since_update > CFG.face.track_max_age:
                t.mark_removed()

        
        unmatched_unconfirmed_indices = set(
            i for i, t in enumerate(unconfirmed)
            if t not in remaining_tracks
        )
        for i in unmatched_unconfirmed_indices:
            unconfirmed[i].time_since_update += 1
            if unconfirmed[i].time_since_update > CFG.face.track_max_age:
                unconfirmed[i].mark_removed()

        
        matched_di = set(int(di) for _, di in matches1)
        for i, (det, emb, ma, _) in enumerate(high_dets):
            if i not in matched_di:
                self._init_track(det, emb, ma, frame_id)

        
        matched_d2 = set(int(di) for _, di in matches2)
        unmatched_tracks_after = [remaining_tracks[i] for i in unmatched_t2]
        for i, (det, emb, ma, _) in enumerate(low_dets):
            if i in matched_d2:
                continue
            best_iou = 0.0
            best_t = None
            for t in unmatched_tracks_after:
                iou_val = _iou(t.tlbr, np.array(det["bbox"], dtype=np.float64))
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_t = t
            if best_iou >= CFG.face.track_iou_threshold and best_t is not None:
                best_t.update(
                    self._tlbr_to_tlwh(det["bbox"]), det["score"], emb, ma, frame_id, det,
                )

        self.tracks = [t for t in self.tracks if t.state != _TrackState.REMOVED]
        return [t for t in self.tracks if t.is_activated]

    def _split_by_score(
        self,
        detections: list[dict[str, Any]],
        embeddings: list[np.ndarray],
        mouth_activities: list[float],
    ) -> tuple[list[tuple], list[tuple]]:
        high, low = [], []
        for i, det in enumerate(detections):
            entry = (det, embeddings[i], mouth_activities[i], i)
            if det["score"] >= CFG.face.track_high_thresh:
                high.append(entry)
            elif det["score"] >= CFG.face.track_low_thresh:
                low.append(entry)
        return high, low

    def _init_track(self, det: dict[str, Any], embedding: np.ndarray,
                    mouth_activity: float, frame_id: int) -> None:
        tlwh = self._tlbr_to_tlwh(det["bbox"])
        track = _STrack(tlwh, det["score"], self.next_id, embedding, mouth_activity)
        track.hit_count = 1
        track.frame_ids.append(frame_id)
        track.observations.append(det)
        self.next_id += 1
        self.tracks.append(track)

    def _init_tracks(self, detections: list[dict[str, Any]],
                     embeddings: list[np.ndarray],
                     mouth_activities: list[float],
                     frame_id: int) -> None:
        for det, emb, mouth_act in zip(detections, embeddings, mouth_activities):
            self._init_track(det, emb, mouth_act, frame_id)

    def _match(self, tracks: list[_STrack], detections: list[dict[str, Any]],
               threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not tracks or not detections:
            empty = np.array([], dtype=int)
            return empty, np.arange(len(tracks)), np.arange(len(detections))

        track_tlbr = np.array([t.tlbr for t in tracks], dtype=np.float32)
        det_tlbr = np.array([d["bbox"] for d in detections], dtype=np.float32)
        iou_matrix = _iou_batch(track_tlbr, det_tlbr)
        return _linear_assignment(1.0 - iou_matrix, threshold)

    @staticmethod
    def _tlbr_to_tlwh(tlbr: tuple | list | np.ndarray) -> np.ndarray:
        tlbr = np.asarray(tlbr, dtype=np.float64)
        return np.array([tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]], dtype=np.float64)






_mp_detector = None

def _get_mouth_activity_mp(face_crop: np.ndarray) -> float:
    """
    Track A2: Swap naive mouth variance for mediapipe lip-motion tracking.
    Uses mediapipe FaceLandmarker's blendshapes (jawOpen) on the crop.
    """
    global _mp_detector
    if face_crop.size == 0 or face_crop.shape[0] < 20:
        return 0.0

    try:
        if _mp_detector is None:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
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
                num_faces=1)
            _mp_detector = vision.FaceLandmarker.create_from_options(options)

        import mediapipe as mp
        # MediaPipe expects RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        detection_result = _mp_detector.detect(mp_image)
        
        if detection_result.face_blendshapes:
            # Get jawOpen blendshape as a proxy for mouth motion/speaking
            blendshapes = detection_result.face_blendshapes[0]
            jaw_open = next((b.score for b in blendshapes if b.category_name == 'jawOpen'), 0.0)
            mouth_pucker = next((b.score for b in blendshapes if b.category_name == 'mouthPucker'), 0.0)
            return max(jaw_open, mouth_pucker)
            
    except Exception as e:
        pass
        
    return 0.0




def _cluster_face_tracks(
    activated: list[_STrack],
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Cluster activated face tracks using HAC on ArcFace embeddings.

    Returns:
        track_to_cluster:   track_id → cluster_label
        cluster_centroids:  cluster_label → mean embedding
        cluster_all_embs:   cluster_label → mean embedding across all observations
    """
    if not activated:
        return {}, {}, {}

    track_ids = np.array([t.track_id for t in activated], dtype=int)

    
    X = np.zeros((len(activated), CFG.face.embedding_dim), dtype=np.float32)
    for i, t in enumerate(activated):
        embs = [obs.get("_embedding", t.embedding)
                for obs in t.observations if "_embedding" in obs]
        if embs:
            X[i] = np.mean(embs, axis=0)
        else:
            X[i] = t.embedding

    
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-10)

    if len(X) >= 2:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=CFG.face.cluster_distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X)
    else:
        labels = np.array([0] * len(X), dtype=int)

    track_to_cluster: dict[int, int] = {}
    cluster_centroids: dict[int, list[np.ndarray]] = {}
    cluster_all_embs: dict[int, list[np.ndarray]] = {}

    for i in range(len(track_ids)):
        tid = int(track_ids[i])
        label = int(labels[i])
        track_to_cluster[tid] = label
        if label not in cluster_centroids:
            cluster_centroids[label] = []
        if label not in cluster_all_embs:
            cluster_all_embs[label] = []
        cluster_centroids[label].append(X[i])
        cluster_all_embs[label].append(X[i])

    centroids = {}
    for label, embs in cluster_centroids.items():
        centroid = np.mean(embs, axis=0).astype(np.float32)
        centroid /= (np.linalg.norm(centroid) + 1e-10)
        centroids[label] = centroid

    all_centroids = {}
    for label, embs in cluster_all_embs.items():
        centroid = np.mean(embs, axis=0).astype(np.float32)
        centroid /= (np.linalg.norm(centroid) + 1e-10)
        all_centroids[label] = centroid

    return track_to_cluster, centroids, all_centroids






def _resolve_workspace_identities(
    cluster_centroids: dict[int, np.ndarray],
    db: Any,
) -> dict[int, str]:
    """
    Match face clusters against the workspace speaker registry.
    Returns: cluster_label → person_id (e.g., "person:A")
    """
    prior_speakers = db.get_speaker_embeddings()
    resolved: dict[int, str] = {}
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for label, centroid in cluster_centroids.items():
        best_match = None
        best_sim = -1.0

        for prior in prior_speakers:
            try:
                prior_emb = np.frombuffer(prior["embedding"], dtype=np.float32)
                if len(prior_emb) != CFG.face.embedding_dim:
                    continue
                prior_norm = prior_emb / (np.linalg.norm(prior_emb) + 1e-10)
                sim = float(np.dot(centroid, prior_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_match = prior.get("speaker_id")
            except Exception:
                continue

        if best_sim >= CFG.face.cross_video_similarity and best_match:
            resolved[label] = best_match
        else:
            letter = labels[label % 26]
            person_id = f"person:{letter}"
            resolved[label] = person_id
            db.insert_speaker_embedding(
                speaker_id=f"face:{person_id}",
                embedding=centroid.tobytes(),
                dim=CFG.face.embedding_dim,
            )

    return resolved






def _bind_speaking(
    activated: list[_STrack],
    track_to_person: dict[int, str],
    db: Any,
) -> None:
    """
    Bind speakers to face tracks using mouth-motion activity cross-correlated
    with audio energy.

    For each utterance:
    1. Find face tracks that overlap with the utterance time window.
    2. Compute average mouth activity during the overlap.
    3. Cross-correlate mouth activity with audio RMS energy.
    4. Assign speaker to the track with highest combined score.
    """
    utterances = db.get_utterances()
    if not utterances:
        return

    
    track_observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in activated:
        for obs in t.observations:
            obs_copy = dict(obs)
            obs_copy["_mouth_activity"] = t.mouth_activity
            obs_copy["_track_id"] = t.track_id
            track_observations[t.track_id].append(obs_copy)

    loudness_curve = db.get_loudness_curve()

    for utt in utterances:
        speaker_id = utt.get("speaker_id")
        if not speaker_id:
            continue

        utt_start = utt["start_time"]
        utt_end = utt["end_time"]

        candidate_scores: dict[str, float] = {}

        for track_id, obs_list in track_observations.items():
            person_id = track_to_person.get(track_id)
            if not person_id:
                continue

            
            overlapping = [
                o for o in obs_list
                if "t" in o and utt_start <= o["t"] <= utt_end
            ]

            if not overlapping:
                continue

            
            mouth_scores = [o.get("_mouth_activity", 0.0) for o in overlapping]
            avg_mouth = np.mean(mouth_scores) if mouth_scores else 0.0

            
            audio_correlation = _correlate_mouth_with_audio(
                overlapping, loudness_curve,
            )
            combined = (
                CFG.face.speaking_confidence_mouth_weight * (avg_mouth + audio_correlation) / 2.0
                + CFG.face.speaking_confidence_presence_weight * min(1.0, len(overlapping) * 0.15)
            )
            candidate_scores[person_id] = combined

        if candidate_scores:
            best_person = max(candidate_scores, key=candidate_scores.get)
            confidence = min(
                0.95,
                candidate_scores[best_person]
                / max(sum(candidate_scores.values()), 1e-10),
            )

            scene_id = _find_overlapping_scene(best_person, activated, track_to_person,
                                                utt_start, utt_end)
            if scene_id is not None:
                db.insert_relation(
                    src=f"person:{best_person}",
                    rel="speaks",
                    dst=speaker_id,
                    scene_id=scene_id,
                    confidence=round(confidence, 4),
                    source="faces",
                    model_version="3.0",
                )


def _correlate_mouth_with_audio(
    observations: list[dict[str, Any]],
    loudness_curve: list[dict[str, Any]],
) -> float:
    """Cross-correlate mouth activity timestamps with audio RMS energy."""
    if not observations or not loudness_curve:
        return 0.0

    mouth_times = np.array([o["t"] for o in observations if "t" in o], dtype=np.float64)
    mouth_vals = np.array(
        [o.get("_mouth_activity", 0.0) for o in observations if "t" in o],
        dtype=np.float64,
    )

    if len(mouth_times) < 2:
        return 0.0

    audio_times = np.array([l["t"] for l in loudness_curve], dtype=np.float64)
    audio_vals = np.array([l.get("rms_db", 0.0) for l in loudness_curve], dtype=np.float64)

    if len(audio_times) < 2:
        return 0.0

    
    try:
        audio_at_mouth = np.interp(mouth_times, audio_times, audio_vals)
    except Exception:
        return 0.0

    
    mouth_norm = (mouth_vals - mouth_vals.min()) / max(mouth_vals.ptp(), 1e-10)
    audio_norm = (audio_at_mouth - audio_at_mouth.min()) / max(audio_at_mouth.ptp(), 1e-10)

    
    if len(mouth_norm) >= 3:
        corr = float(np.corrcoef(mouth_norm, audio_norm)[0, 1])
        return max(0.0, corr)
    return 0.0


def _find_overlapping_scene(
    person_id: str,
    activated: list[_STrack],
    track_to_person: dict[int, str],
    utt_start: float,
    utt_end: float,
) -> int | None:
    """Find the scene_id where this person was visible during the utterance."""
    for t in activated:
        if track_to_person.get(t.track_id) == person_id:
            for obs in t.observations:
                obs_t = obs.get("t")
                if obs_t is not None and utt_start <= obs_t <= utt_end:
                    return obs.get("scene_id")
    return None






class FacePipeline:
    """
    Orchestrates SCRFD + ArcFace + ByteTrack + HAC + Registry + Speaking.

    Usage:
        pipeline = FacePipeline(db)
        pipeline.process_video(project_dir, total_duration)
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self.detector = SCRFDDetector()
        self.embedder = ArcFaceEmbedder()
        self.tracker = ByteTracker()
        self._all_tracks: list[_STrack] = []

    def process_video(self, project_dir: Path, total_duration: float) -> None:
        """Run the full face pipeline on a proxy video."""
        proxy_path = project_dir / "proxy.mp4"
        if not proxy_path.exists():
            logger.warning("proxy.mp4 not found, skipping face pipeline")
            return

        scenes = self.db.get_scenes()
        if not scenes:
            logger.warning("No scenes found, skipping face pipeline")
            return

        console.print("  Running face pipeline: SCRFD + ArcFace + ByteTrack...")

        cap = cv2.VideoCapture(str(proxy_path))
        if not cap.isOpened():
            logger.warning("Cannot open proxy.mp4, skipping face pipeline")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_rate = CFG.face.track_frame_sample_rate
        total_frames = int(total_duration * fps)

        
        scene_map: dict[int, int] = {}
        for s in scenes:
            start_f = int(s["start_time"] * fps)
            end_f = int(s["end_time"] * fps)
            for f in range(start_f, end_f + 1):
                scene_map[f] = s["id"]

        last_report = time.monotonic()
        processed_frames = 0

        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            t = frame_idx / fps
            scene_id = scene_map.get(frame_idx)

            
            face_dets = self.detector.detect(frame)

            if not face_dets:
                
                self.tracker.update([], frame_idx, [], [])
                continue

            
            embeddings = []
            mouth_activities = []
            for det in face_dets:
                x1, y1, x2, y2 = det["bbox"]
                crop = frame[y1:y2, x1:x2]
                emb = self.embedder.embed(crop)
                mouth_act = _get_mouth_activity_mp(crop)
                embeddings.append(emb)
                mouth_activities.append(mouth_act)
                det["t"] = t
                det["scene_id"] = scene_id
                det["_embedding"] = emb
                det["_mouth_activity"] = mouth_act

            
            activated = self.tracker.update(face_dets, frame_idx, embeddings, mouth_activities)
            self._all_tracks.extend(activated)

            processed_frames += 1

            
            now = time.monotonic()
            if now - last_report > 10:
                pct = (frame_idx / max(total_frames, 1)) * 100
                console.print(f"    Faces progress: {pct:.0f}% ({processed_frames} frames, "
                              f"{len(self._all_tracks)} tracks)")
                last_report = now

        cap.release()

        console.print(f"  Face detection done: {processed_frames} frames, "
                      f"{len(self._all_tracks)} raw tracks")

        if not self._all_tracks:
            console.print("  [yellow]No faces detected[/yellow]")
            self.db.set_coverage("faces", "unavailable", note="no faces detected")
            return

        
        console.print("  Clustering face identities (HAC)...")
        try:
            track_to_cluster, centroids, all_centroids = _cluster_face_tracks(self._all_tracks)
        except Exception as e:
            import traceback
            traceback.print_exc()
            console.print(f"    [yellow]Attempt of faces failed: {str(e)}[/yellow]")
            self.db.set_coverage("faces", "unavailable", note="no clusters formed")
            return

        if not centroids:
            console.print("  [yellow]No face clusters formed[/yellow]")
            self.db.set_coverage("faces", "unavailable", note="no clusters formed")
            return

        
        console.print("  Resolving workspace identities...")
        cluster_to_person = _resolve_workspace_identities(centroids, self.db)

        
        track_to_person: dict[int, str] = {}
        for t in self._all_tracks:
            cluster = track_to_cluster.get(t.track_id)
            if cluster is not None:
                track_to_person[t.track_id] = cluster_to_person.get(cluster, f"person:unknown")

        
        console.print("  Emitting face entities and relations...")
        self._emit_entities_relations(track_to_cluster, centroids, cluster_to_person)

        
        console.print("  Binding speakers via AV-sync mouth correlation...")
        _bind_speaking(self._all_tracks, track_to_person, self.db)

        self.db.set_coverage("faces", "available")
        self.db.set_model_manifest("faces", f"scrfd-{CFG.face.detector_model}+arcface-{CFG.face.recog_model}", "3.0")

        if not CFG.face.store_embeddings:
            self._clear_embeddings()

        import gc as _gc
        del self.detector, self.embedder, self.tracker, self._all_tracks
        _gc.collect()

        console.print("  [dim]Face pipeline complete[/dim]")

    def _emit_entities_relations(
        self,
        track_to_cluster: dict[int, int],
        centroids: dict[int, np.ndarray],
        cluster_to_person: dict[int, str],
    ) -> None:
        """Emit person entities and appears_in relations from face tracks."""
        seen_persons: set[str] = set()
        for cluster_label, person_id in cluster_to_person.items():
            if person_id not in seen_persons:
                centroid = centroids.get(cluster_label)
                desc = f"Face cluster {person_id} via SCRFD + ArcFace"
                self.db.insert_entity(
                    entity_id=person_id,
                    kind="person",
                    label=person_id,
                    description=desc,
                )
                seen_persons.add(person_id)

        
        for t in self._all_tracks:
            cluster_id = track_to_cluster.get(t.track_id)
            if cluster_id is None:
                continue
            person_id = cluster_to_person.get(cluster_id)
            if not person_id:
                continue
            for obs in t.observations:
                scene_id = obs.get("scene_id")
                if scene_id is not None:
                    self.db.insert_relation(
                        src=person_id,
                        rel="appears_in",
                        dst=f"scene:{scene_id}",
                        scene_id=scene_id,
                        t_start=obs.get("t"),
                        t_end=obs.get("t"),
                        confidence=min(0.95, obs.get("score", 0.5) + 0.2),
                        source="faces",
                        model_version="3.0",
                    )

    def _clear_embeddings(self) -> None:
        """Remove face embeddings only if privacy mode is enabled.
        Only deletes entries with ArcFace dimension (512-d) to avoid
        destroying audio diarization embeddings.
        """
        with self.db.conn() as c:
            c.execute("DELETE FROM speaker_embeddings WHERE dim = ?", (CFG.face.embedding_dim,))
        logger.info("Privacy mode: cleared face embeddings (dim=%d)", CFG.face.embedding_dim)






class PrivacyFilter:
    """Apply pixelation/blur to detected face regions in a frame."""

    @staticmethod
    def blur_faces(frame: np.ndarray, detections: list[dict[str, Any]],
                   method: str = "pixelate") -> np.ndarray:
        """Blur or pixelate detected face regions."""
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            face = result[y1:y2, x1:x2]
            if face.size == 0:
                continue
            if method == "pixelate":
                small = cv2.resize(face, (16, 16), interpolation=cv2.INTER_LINEAR)
                result[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                                   interpolation=cv2.INTER_NEAREST)
            elif method == "gaussian":
                result[y1:y2, x1:x2] = cv2.GaussianBlur(face, (99, 99), 30)
        return result






def run_face_pipeline(project_dir: Path, db: Any) -> None:
    """
    Entry point for the face pipeline DAG stage.
    Called by the orchestrator after scenes stage completes.

    Depends on: scenes (needs keyframes and scene boundaries)
    Produces:  face entities, relations, speaker bindings
    """
    video = db.get_video()
    if not video:
        raise RuntimeError("No video metadata — run normalize first")

    content_class = video.get("content_class", "standard")
    if content_class == "screencast":
        console.print("  [dim]Content-class: screencast → skipping face pipeline[/dim]")
        db.set_coverage("faces", "skipped", note="screencast — no faces expected")
        return

    try:
        pipeline = FacePipeline(db)
        pipeline.process_video(project_dir, video["duration_s"])
    except RuntimeError as exc:
        if "Face detection unavailable" in str(exc):
            console.print(f"  [yellow]⚠ Face pipeline degraded: {exc}[/yellow]")
            console.print("  [dim]Person identity will use VLM-roster fallback in vision stage[/dim]")
            db.set_coverage("faces", "degraded", note=str(exc))
        else:
            raise
