"""
Hybrid Retrieval Engine (§5.2) — four channels, then fusion.

Channels:
1. Structured (graph/SQL) — exact entity/relation matches
2. Keyword (BM25) — transcript + caption keyword search
3. Vector (FAISS) — MiniLM semantic + CLIP visual similarity
4. Metadata (SQL) — time ranges, shot types, is_broll, story roles

Fusion: weighted reciprocal-rank fusion (structured 3×, metadata 2×, keyword 1×, vector 1×)
"""

from __future__ import annotations

import os
import pickle
import re
import difflib
import gc
from pathlib import Path
from collections import defaultdict

# Prevent OpenMP duplicate-library crash on macOS (torch vs opencv)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.schemas import (
    EditIntent, Evidence, RetrievalResult, Segment,
)

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

console = Console()


_cross_encoder_instance = None

def _get_cross_encoder():
    """Lazy-load a lightweight cross-encoder for re-ranking retrieval results."""
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    return _cross_encoder_instance





def _structured_search(query: str, modalities: list[str], db: ProjectDB) -> list[Segment]:
    """Exact matches via graph traversal and metadata filters."""
    results: list[Segment] = []
    query_lower = query.lower()

    
    if any(kw in query_lower for kw in ["silence", "pause", "dead time", "gap"]):
        silences = db.get_silences()
        for s in silences:
            results.append(Segment(
                start=s["start_time"], end=s["end_time"],
                scene_ids=[], score=1.0,
                evidence=[Evidence(source="vad", detail=f"silence {s['duration']:.1f}s", t=s["start_time"])],
            ))

    
    if any(kw in query_lower for kw in ["filler", "um", "uh", "hmm"]):
        fillers = db.get_fillers()
        for f in fillers:
            results.append(Segment(
                start=f["start_time"], end=f["end_time"],
                scene_ids=[], score=f["confidence"],
                evidence=[Evidence(source="filler_detector", detail=f"filler: '{f['word']}'", t=f["start_time"])],
            ))

    
    if any(kw in query_lower for kw in ["retake", "mistake", "final take", "repeated"]):
        clusters = db.get_retake_clusters()
        for cluster_id, members in clusters.items():
            if "final" in query_lower or "last" in query_lower:
                non_final = members[-1:]
            else:
                non_final = members[:-1]

            for m in non_final:
                utt = db.get_utterances()
                matching_utt = [u for u in utt if u["id"] == m["utt_id"]]
                if matching_utt:
                    u = matching_utt[0]
                    results.append(Segment(
                        start=u["start_time"], end=u["end_time"],
                        scene_ids=[], score=0.9,
                        evidence=[Evidence(
                            source="retake_detector",
                            detail=f"retake {m['take_index']+1} of cluster {cluster_id}",
                            t=u["start_time"],
                        )],
                    ))

    
    if "person" in modalities or any(kw in query_lower for kw in ["person", "who", "speaker", "speaking"]):
        from trim_engine.ingest.graph import scenes_where

        entities = db.get_entities(kind="person")
        for entity in entities:
            label = entity["label"].lower()
            entity_id = entity["id"]

            if label in query_lower or entity_id.lower() in query_lower:
                matching_scenes = scenes_where(db, person=entity_id)
                for scene in matching_scenes:
                    results.append(Segment(
                        start=scene["start_time"], end=scene["end_time"],
                        scene_ids=[scene["id"]], score=1.0,
                        evidence=[Evidence(
                            source="graph", detail=f"{entity_id} appears",
                            t=scene["start_time"],
                        )],
                    ))

            if entity.get("is_owner") and any(kw in query_lower for kw in ["me", "i'm", "my ", "i "]):
                matching_scenes = scenes_where(db, person=entity_id)
                for scene in matching_scenes:
                    results.append(Segment(
                        start=scene["start_time"], end=scene["end_time"],
                        scene_ids=[scene["id"]], score=1.0,
                        evidence=[Evidence(source="graph", detail="owner appears", t=scene["start_time"])],
                    ))

    
    if "location" in modalities:
        scenes = db.get_scenes()
        for scene in scenes:
            if scene.get("location") and scene["location"].lower() in query_lower:
                results.append(Segment(
                    start=scene["start_time"], end=scene["end_time"],
                    scene_ids=[scene["id"]], score=1.0,
                    evidence=[Evidence(
                        source="graph", detail=f"location: {scene['location']}",
                        t=scene["start_time"],
                    )],
                ))

    # §1.2 Audio-events channel: laughter, clapping, applause
    _audio_event_map = {
        "laugh": "laughter", "laughing": "laughter", "laughter": "laughter",
        "chuckle": "laughter", "giggling": "laughter",
        "clap": "applause", "clapping": "applause", "applause": "applause",
        "cheer": "applause", "cheering": "applause",
        "gasp": "gasp", "gasping": "gasp",
    }
    for kw, event_type in _audio_event_map.items():
        if kw in query_lower:
            events = db.get_audio_events(event_type=event_type)
            for ev in events:
                score = ev.get("confidence", 0.8) or 0.8
                results.append(Segment(
                    start=ev["start_time"], end=ev["end_time"],
                    scene_ids=[], score=score,
                    evidence=[Evidence(
                        source="audio_events",
                        detail=f"audio event: {event_type} (conf={score:.2f})",
                        t=ev["start_time"],
                    )],
                ))
            break  # only match the first keyword

    # §1.3 Object relations channel: "product visible", "phone in hand", "whiteboard"
    if "object" in modalities or any(kw in query_lower for kw in ["product", "phone", "laptop", "whiteboard", "cup", "book", "camera", "screen", "mic", "microphone"]):
        obj_entities = db.get_entities(kind="object")
        if obj_entities:
            distinct_labels = list(set(e["label"].lower() for e in obj_entities))
            query_words = query_lower.split()
            matched_labels = set()
            for label in distinct_labels:
                for word in query_words:
                    if len(word) >= 3 and (word in label or label in word):
                        matched_labels.add(label)
            # Fuzzy fallback via difflib
            if not matched_labels:
                for word in query_words:
                    if len(word) >= 3:
                        close = difflib.get_close_matches(word, distinct_labels, n=2, cutoff=0.6)
                        matched_labels.update(close)
            for label in matched_labels:
                # Find scenes containing this object via the relations table
                rels = db.get_relations(rel="contains")
                for rel in rels:
                    obj_entity = db.get_entity(rel["dst"])
                    if obj_entity and obj_entity["label"].lower() == label and rel.get("scene_id"):
                        scene = db.get_scene(rel["scene_id"])
                        if scene:
                            results.append(Segment(
                                start=scene["start_time"], end=scene["end_time"],
                                scene_ids=[scene["id"]], score=rel.get("confidence", 0.7) or 0.7,
                                evidence=[Evidence(
                                    source="graph_object",
                                    detail=f"object '{label}' in scene (rel=contains)",
                                    t=scene["start_time"],
                                )],
                            ))

    # §1.4 Emotion relations channel: "emotional moments", "funniest"
    _emotion_vocab = {
        "happy": ["happy", "joy", "joyful"], "excited": ["excited", "excitement"],
        "sad": ["sad", "sadness", "melancholy"], "angry": ["angry", "anger", "frustrated"],
        "tense": ["tense", "tension", "nervous"], "emotional": ["emotional", "emotions"],
        "funny": ["funny", "funniest", "humorous", "comedy", "hilarious"],
    }
    for category, keywords in _emotion_vocab.items():
        if any(kw in query_lower for kw in keywords):
            scenes = db.get_scenes()
            for scene in scenes:
                if scene.get("emotion_label") and scene.get("emotion_intensity"):
                    label_lower = scene["emotion_label"].lower()
                    intensity = scene["emotion_intensity"]
                    # Match: funny → positive emotions + high intensity
                    match = False
                    if category == "funny" and label_lower in ("happy", "joy", "amused", "excited") and intensity > 0.5:
                        match = True
                    elif category == "emotional" and intensity > 0.6:
                        match = True
                    elif any(kw in label_lower for kw in keywords):
                        match = True
                    if match:
                        results.append(Segment(
                            start=scene["start_time"], end=scene["end_time"],
                            scene_ids=[scene["id"]], score=intensity,
                            evidence=[Evidence(
                                source="emotion",
                                detail=f"emotion: {scene['emotion_label']} (intensity={intensity:.2f})",
                                t=scene["start_time"],
                            )],
                        ))
            break

    # §3.3 Derived moments channel: funny, awkward, applause
    _derived_map = {
        "awkward": "awkward", "cringe": "awkward", "uncomfortable": "awkward",
        "funniest": "funny", "funny": "funny", "hilarious": "funny",
        "applause": "applause",
    }
    for kw, kind in _derived_map.items():
        if kw in query_lower:
            moments = db.get_derived_moments(kind=kind)
            for m in moments:
                results.append(Segment(
                    start=m["start_time"], end=m["end_time"],
                    scene_ids=[], score=m.get("confidence", 0.7) or 0.7,
                    evidence=[Evidence(
                        source="derived_moment",
                        detail=f"derived moment: {kind} ({m['formula']})",
                        t=m["start_time"],
                    )],
                ))
            break

    # §3.2 Dialogue-act channel: "keep only questions" / "keep only answers"
    if any(kw in query_lower for kw in ["question", "questions", "answer", "answers", "q&a"]):
        utts = db.get_utterances()
        target_act = None
        if any(kw in query_lower for kw in ["question", "questions"]):
            target_act = "question"
        elif any(kw in query_lower for kw in ["answer", "answers"]):
            target_act = "answer"
        if target_act:
            for u in utts:
                if u.get("dialogue_act") == target_act:
                    results.append(Segment(
                        start=u["start_time"], end=u["end_time"],
                        scene_ids=[], score=0.9,
                        evidence=[Evidence(
                            source="dialogue_act",
                            detail=f"utterance act={target_act}: '{u['text'][:60]}'",
                            t=u["start_time"],
                        )],
                    ))

    # §3.4 Off-topic detection
    if any(kw in query_lower for kw in ["off-topic", "offtopic", "off topic", "tangent", "unrelated"]):
        topics = db.get_topics()
        if topics:
            # Main topic = longest total duration
            topic_durations: dict[str, float] = defaultdict(float)
            for t in topics:
                topic_durations[t["label"]] += t["end_time"] - t["start_time"]
            main_topic = max(topic_durations, key=topic_durations.get)
            for t in topics:
                if t["label"] != main_topic:
                    results.append(Segment(
                        start=t["start_time"], end=t["end_time"],
                        scene_ids=[], score=0.8,
                        evidence=[Evidence(
                            source="topic_model",
                            detail=f"off-topic: '{t['label']}' (main='{main_topic}')",
                            t=t["start_time"],
                        )],
                    ))

    
    topics = db.get_topics()
    for topic in topics:
        label_lower = topic.get("label", "").lower()
        t_class = topic.get("class", "").lower()
        
        match = False
        if label_lower and len(label_lower) >= 3:
            if label_lower in query_lower or query_lower in label_lower:
                match = True
        
        if t_class and t_class != "other" and len(t_class) >= 3:
            if t_class in query_lower:
                match = True
                
        if match:
            results.append(Segment(
                start=topic["start_time"], end=topic["end_time"],
                scene_ids=[], score=0.95,
                evidence=[Evidence(
                    source="topic_model",
                    detail=f"topic={topic.get('label')} (class={topic.get('class')})",
                    t=topic["start_time"],
                )],
            ))

    import re
    ts_pattern = r'(?:from\s+)?(?:(\d+):)?(\d+)\s*(?:s|sec|second|seconds|m|min|minute|minutes)?\s*(?:to|-|and)\s*(?:(\d+):)?(\d+)\s*(?:s|sec|second|seconds|m|min|minute|minutes|nd|th|st|rd)?'
    match = re.search(ts_pattern, query_lower)
    if match and any(kw in query_lower for kw in ["second", "minute", "s", "m", ":", "-", "to", "and", "from", "nd", "th"]):
        s_min = int(match.group(1)) if match.group(1) else 0
        s_sec = int(match.group(2))
        e_min = int(match.group(3)) if match.group(3) else 0
        e_sec = int(match.group(4))
        start_ts = float(s_min * 60 + s_sec)
        end_ts = float(e_min * 60 + e_sec)
        if end_ts > start_ts:
            results.append(Segment(
                start=start_ts, end=end_ts,
                scene_ids=[], score=1.0,
                evidence=[Evidence(
                    source="explicit_timestamp",
                    detail=f"explicit timestamp {start_ts}s to {end_ts}s",
                    t=start_ts,
                )],
            ))

    return results





def _keyword_search(query: str, project_dir: Path, db: ProjectDB, top_k: int = 20) -> list[Segment]:
    """BM25 keyword search over transcript + captions."""
    bm25_path = project_dir / "faiss" / "bm25.pkl"
    if not bm25_path.exists():
        return []

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    doc_refs = data["doc_refs"]

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]
    utts = db.get_utterances()
    utt_text_map = {u["id"]: u["text"] for u in utts}
    scenes = db.get_scenes()
    scene_text_map = {s["id"]: " ".join(u["text"] for u in utts if u["start_time"] >= s["start_time"] and u["end_time"] <= s["end_time"]) for s in scenes}

    results: list[Segment] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break

        ref = doc_refs[idx]
        if ref["type"] == "utterance":
            match_text = utt_text_map.get(ref["id"], "")
        else:
            match_text = scene_text_map.get(ref["id"], "")

        results.append(Segment(
            start=ref["start"], end=ref["end"],
            scene_ids=[ref["id"]] if ref["type"] == "scene" else [],
            score=min(score / 10.0, 1.0),
            evidence=[Evidence(
                source="keyword",
                detail=f"BM25 match: '{match_text}'",
                t=ref["start"],
            )],
        ))

    if not results and any(w[0].isupper() for w in tokens):
        words = db.get_words()
        unique_words = list(set(w["word"] for w in words))
        for token in tokens:
            matches = difflib.get_close_matches(token, unique_words, n=3, cutoff=0.7)
            if matches:
                utts = db.get_utterances()
                for u in utts:
                    if any(re.search(rf"\b{re.escape(m)}\b", u["text"], re.IGNORECASE) for m in matches):
                        results.append(Segment(
                            start=u["start_time"], end=u["end_time"],
                            scene_ids=[], score=0.6,
                            evidence=[Evidence(source="keyword_fuzzy", detail=f"Fuzzy match '{matches[0]}' for '{token}'", t=u["start_time"])],
                        ))

    return results





def _vector_search(query: str, project_dir: Path, db: ProjectDB, top_k: int = 20) -> list[Segment]:
    """Semantic vector search via FAISS (Text and CLIP channels)."""
    results: list[Segment] = []

    
    global _sentence_transformers_unavailable
    if not globals().get("_sentence_transformers_unavailable"):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            text_index_path = project_dir / "faiss" / "scene_text.index"

            if not text_index_path.exists():
                from trim_engine.ingest.index import run_index_builder
                run_index_builder(project_dir, db)

            if text_index_path.exists():
                global _text_model_instance
                if '_text_model_instance' not in globals():
                    _text_model_instance = SentenceTransformer(CFG.embedding.text_model_name)
            
            model = _text_model_instance
            query_emb = model.encode([query], normalize_embeddings=True)

            index = faiss.read_index(str(text_index_path))
            scores, indices = index.search(query_emb.astype(np.float32), min(top_k, index.ntotal))

            scenes = db.get_scenes()
            utts = db.get_utterances()
            scene_text_map = {s["id"]: " ".join(u["text"] for u in utts if u["start_time"] >= s["start_time"] and u["end_time"] <= s["end_time"]) for s in scenes}

            best_score = float(max(scores[0])) if len(scores[0]) > 0 else 0.0
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or float(score) < max(0.15, best_score - 0.15):
                    continue
                scene_id = db.get_scene_id_for_vector(int(idx) + len(scenes), "text")
                if scene_id is not None:
                    scene = db.get_scene(scene_id)
                    if scene:
                        match_text = scene_text_map.get(scene_id, "")
                        results.append(Segment(
                            start=scene["start_time"], end=scene["end_time"],
                            scene_ids=[scene_id], score=float(score),
                            evidence=[Evidence(source="vector", detail=f"semantic text match: '{match_text}'", t=scene["start_time"])],
                        ))
        except Exception as e:
            console.print(f"    [yellow]Text vector search failed: {e}[/yellow]")
            global _sentence_transformers_unavailable
            _sentence_transformers_unavailable = True
        finally:
            gc.collect()

    
    
    
    # §1.8 Fix: removed the .dylibs check that silently disabled CLIP on macOS.
    # KMP_DUPLICATE_LIB_OK is set at module top to prevent OpenMP crashes.
    _skip_clip = os.environ.get("TRIM_SKIP_CLIP_RETRIEVAL", "0") == "1"

    if not _skip_clip:
        try:
            import faiss
            clip_index_path = project_dir / "faiss" / "scene_clip.index"

            if clip_index_path.exists():
                import torch
                import open_clip

                global _clip_model_instance, _clip_tokenizer_instance
                if '_clip_model_instance' not in globals():
                    _clip_model_instance, _, _ = open_clip.create_model_and_transforms(CFG.embedding.clip_model_name, pretrained=CFG.embedding.clip_pretrained)
                    _clip_model_instance.eval()
                    _clip_tokenizer_instance = open_clip.get_tokenizer(CFG.embedding.clip_model_name)
                
                model = _clip_model_instance
                tokenizer = _clip_tokenizer_instance
                
                try:
                    text_tokens = tokenizer([query])
                except AttributeError:
                    # Fallback for newer transformers with T5Tokenizer
                    ctx_len = getattr(model, "context_length", 64)
                    text_tokens = tokenizer.tokenizer([query], padding="max_length", truncation=True, max_length=ctx_len, return_tensors="pt")["input_ids"]

                with torch.no_grad():
                    query_emb = model.encode_text(text_tokens)
                    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
                    query_emb_np = query_emb.cpu().numpy().astype(np.float32)

                index = faiss.read_index(str(clip_index_path))
                scores, indices = index.search(query_emb_np, min(top_k, index.ntotal))

                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or score < 0.1:
                        continue
                    scene_id = db.get_scene_id_for_vector(int(idx), "clip")
                    if scene_id is not None:
                        scene = db.get_scene(scene_id)
                        if scene:
                            results.append(Segment(
                                start=scene["start_time"], end=scene["end_time"],
                                scene_ids=[scene_id], score=float(score),
                                evidence=[Evidence(source="vector", detail=f"visual CLIP match (score={score:.2f})", t=scene["start_time"])],
                            ))
                
                # §4.4: Also search per-frame CLIP index (backstop for mid-shot objects/actions)
                frame_idx_path = project_dir / "faiss" / "frame_clip.index"
                frame_map_path = project_dir / "faiss" / "frame_lut_map.pkl"
                if frame_idx_path.exists() and frame_map_path.exists():
                    import pickle
                    with open(frame_map_path, "rb") as f:
                        frame_map = pickle.load(f)
                    f_index = faiss.read_index(str(frame_idx_path))
                    f_scores, f_indices = f_index.search(query_emb_np, min(top_k * 2, f_index.ntotal))
                    for f_score, f_idx in zip(f_scores[0], f_indices[0]):
                        if f_idx < 0 or f_score < 0.15:
                            continue
                        if int(f_idx) < len(frame_map):
                            scene_id = frame_map[int(f_idx)]["scene_id"]
                            scene = db.get_scene(scene_id)
                            if scene:
                                results.append(Segment(
                                    start=scene["start_time"], end=scene["end_time"],
                                    scene_ids=[scene_id], score=float(f_score),
                                    evidence=[Evidence(source="vector", detail=f"frame CLIP match (score={f_score:.2f})", t=scene["start_time"])],
                                ))
                                
                del tokenizer, query_emb, query_emb_np
        except Exception as e:
            import traceback
            console.print(f"    [yellow]CLIP vector search failed or skipped: {type(e).__name__}: {e}\n{traceback.format_exc()}[/yellow]")
        finally:
            gc.collect()

    return results





def _metadata_search(query: str, db: ProjectDB) -> list[Segment]:
    """Retrieves target segments by querying direct scene metadata attributes."""
    results: list[Segment] = []
    query_lower = query.lower()
    scenes = db.get_scenes()

    for scene in scenes:
        
        if any(kw in query_lower for kw in ["b-roll", "broll", "b roll"]):
            if scene.get("is_broll"):
                results.append(Segment(
                    start=scene["start_time"], end=scene["end_time"],
                    scene_ids=[scene["id"]], score=0.9,
                    evidence=[Evidence(source="metadata", detail="is_broll metadata match", t=scene["start_time"])]
                ))

        # §1.1 Indoor/outdoor structured filter
        if any(kw in query_lower for kw in ["outdoor", "outside", "exterior"]):
            if scene.get("indoor") == 0:  # 0 = outdoor
                results.append(Segment(
                    start=scene["start_time"], end=scene["end_time"],
                    scene_ids=[scene["id"]], score=0.95,
                    evidence=[Evidence(source="metadata", detail="indoor=0 (outdoor scene)", t=scene["start_time"])]
                ))
        if any(kw in query_lower for kw in ["indoor", "inside", "interior", "office"]):
            if scene.get("indoor") == 1:  # 1 = indoor
                results.append(Segment(
                    start=scene["start_time"], end=scene["end_time"],
                    scene_ids=[scene["id"]], score=0.95,
                    evidence=[Evidence(source="metadata", detail="indoor=1 (indoor scene)", t=scene["start_time"])]
                ))

        role = scene.get("story_role")
        if role:
            match_roles = [role]
            if role == "intro": match_roles.append("hook")
            if any(r in query_lower for r in match_roles):
                results.append(Segment(
                    start=scene["start_time"], end=scene["end_time"],
                    scene_ids=[scene["id"]], score=0.85,
                    evidence=[Evidence(source="metadata", detail=f"story_role={role} metadata match", t=scene["start_time"])]
                ))

    return results





def _expand_query(query: str) -> list[str]:
    synonyms = {
        "pricing": ["price", "cost", "rates", "subscription", "charge"],
        "laughing": ["laugh", "laughter", "chuckle", "gasp"],
        "awkward": ["pause", "silence", "stumble", "repeat"],
        "sponsor": ["advertisement", "ad", "sponsor", "sponsored"],
        "highlight": ["hook", "climax", "payoff", "highlight"],
    }
    for k, v in synonyms.items():
        if k in query.lower():
            return [query] + [query.replace(k, syn) for syn in v]
    return [query]

def _search_all_channels(
    query: str,
    modalities: list[str],
    project_dir: Path,
    db: ProjectDB,
) -> list[Segment]:
    structured = _structured_search(query, modalities, db)
    keyword = _keyword_search(query, project_dir, db)
    vector = _vector_search(query, project_dir, db)
    metadata = _metadata_search(query, db)

    channel_results = {
        "structured": structured,
        "keyword": keyword,
        "vector": vector,
        "metadata": metadata,
    }

    return _reciprocal_rank_fusion(channel_results, CFG.retrieval.fusion_weights)

def _resolve_compound_intersection(
    query: str,
    modalities: list[str],
    project_dir: Path,
    db: ProjectDB,
) -> list[Segment]:
    parts = []
    if " while " in query.lower():
        parts = query.lower().split(" while ")
    elif " simultaneously " in query.lower():
        parts = query.lower().split(" simultaneously ")

    if len(parts) == 2:
        res_a = []
        for eq_a in _expand_query(parts[0]):
            res_a.extend(_search_all_channels(eq_a, modalities, project_dir, db))
        
        res_b = []
        for eq_b in _expand_query(parts[1]):
            res_b.extend(_search_all_channels(eq_b, modalities, project_dir, db))

        intersections = []
        for sa in res_a:
            for sb in res_b:
                overlap_start = max(sa.start, sb.start)
                overlap_end = min(sa.end, sb.end)
                if overlap_end - overlap_start > 0.2:
                    intersections.append(Segment(
                        start=overlap_start, end=overlap_end,
                        scene_ids=list(set(sa.scene_ids + sb.scene_ids)),
                        score=(sa.score + sb.score) / 2,
                        evidence=sa.evidence + sb.evidence,
                    ))
        if intersections:
            return intersections

    return []





def _reciprocal_rank_fusion(
    channel_results: dict[str, list[Segment]],
    weights: dict[str, float],
    k: int = 60,
    threshold: float = 0.35,
) -> list[Segment]:
    range_scores: dict[tuple[float, float], dict] = {}

    for channel, segments in channel_results.items():
        weight = weights.get(channel, 1.0)
        for rank, seg in enumerate(sorted(segments, key=lambda s: s.score, reverse=True)):
            key = (round(seg.start, 2), round(seg.end, 2))
            rrf_score = weight * (1.0 / (k + rank + 1))

            if key not in range_scores:
                range_scores[key] = {
                    "start": seg.start,
                    "end": seg.end,
                    "scene_ids": set(seg.scene_ids),
                    "score": 0.0,
                    "evidence": [],
                    "channels": set(),
                }
            range_scores[key]["score"] += rrf_score
            range_scores[key]["scene_ids"] |= set(seg.scene_ids)
            range_scores[key]["evidence"].extend(seg.evidence)
            range_scores[key]["channels"].add(channel)

    results: list[Segment] = []
    conf_band = CFG.retrieval.confirmation_band

    for key, data in sorted(range_scores.items(), key=lambda x: x[1]["score"], reverse=True):
        score = data["score"]
        norm_score = min(score * 60, 1.0)

        if norm_score < threshold * 0.5:
            continue

        results.append(Segment(
            start=data["start"],
            end=data["end"],
            scene_ids=sorted(data["scene_ids"]),
            score=norm_score,
            evidence=data["evidence"],
            needs_confirmation=(conf_band[0] <= norm_score <= conf_band[1]),
        ))

    return results

def _merge_adjacent(segments: list[Segment], gap_s: float = 1.0) -> list[Segment]:
    if not segments:
        return segments

    segments = sorted(segments, key=lambda s: s.start)
    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start <= prev.end + gap_s:
            merged[-1] = Segment(
                start=prev.start,
                end=max(prev.end, seg.end),
                scene_ids=sorted(set(prev.scene_ids + seg.scene_ids)),
                score=max(prev.score, seg.score),
                evidence=prev.evidence + seg.evidence,
                needs_confirmation=prev.needs_confirmation or seg.needs_confirmation,
            )
        else:
            merged.append(seg)

    return merged





def retrieve_segments(
    intent: EditIntent,
    db: ProjectDB,
    project_dir: Path,
    retry_count: int = 0,
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []

    
    with db.conn() as c:
        rows = c.execute("SELECT scene_id FROM relations WHERE needs_verification = 1").fetchall()
        needs_verif_ids = {r["scene_id"] for r in rows if r["scene_id"] is not None}

    
    threshold = max(0.15, CFG.retrieval.score_threshold - (0.05 * retry_count))

    # Check if the video has speech transcript
    words = db.get_words()
    has_speech = (len(words) > 0)

    for i, op in enumerate(intent.operations):
        query = op.target.query
        modalities = op.target.modality
        anchor = getattr(op.target, 'anchor', None)

        # Skip speech-specific operations (like filler words or silence) in videos without speech
        query_lower = query.lower()
        if not has_speech and ("filler" in query_lower or "silence" in query_lower or "speech" in query_lower or "voice" in query_lower or "word" in query_lower or "sound" in query_lower or "talk" in query_lower or "speak" in query_lower):
            console.print(f"  [dim]Skipping op[{i}]: \"{query}\" (reason: video contains no speech transcript)[/dim]")
            results.append(RetrievalResult(operation_index=i, segments=[]))
            continue

        console.print(f"  [dim]Retrieving for op[{i}]: \"{query}\" (threshold={threshold:.2f})[/dim]")

        # §3.1: Resolve temporal anchor to a time range filter
        anchor_start, anchor_end = 0.0, float('inf')
        
        # Support both BaseModel (Pydantic v1/v2) and raw dicts
        if anchor and hasattr(anchor, "model_dump"):
            anchor = anchor.model_dump()
        elif anchor and hasattr(anchor, "dict"):
            anchor = anchor.dict()
            
        if anchor and isinstance(anchor, dict):
            anchor_type = anchor.get("type")
            if anchor_type == "absolute":
                anchor_start = anchor.get("start_s", 0.0)
                anchor_end = anchor.get("end_s", float('inf'))
                console.print(f"    [dim]Anchor: absolute → [{anchor_start:.1f}s, {anchor_end:.1f}s][/dim]")
                fused = [Segment(start=anchor_start, end=anchor_end, score=1.0, metadata={"source": "absolute_anchor", "query": query})]
                results.append(RetrievalResult(operation_index=i, segments=fused))
                continue
            
            subject_q = anchor.get("subject_query", "")
            if subject_q:
                from trim_engine.ingest.graph import first_appearance, last_appearance
                # Try to resolve as a person
                t_first = first_appearance(db, subject_q)
                t_last = last_appearance(db, subject_q)
                # Fallback: resolve via structured/vector search
                if t_first is None:
                    sub_segs = _structured_search(subject_q, ["person", "visual"], db)
                    if not sub_segs:
                        sub_segs = _vector_search(subject_q, project_dir, db)
                    if sub_segs:
                        t_first = min(s.start for s in sub_segs)
                        t_last = max(s.end for s in sub_segs)
                if t_first is not None:
                    if anchor_type == "before":
                        anchor_end = t_first
                        console.print(f"    [dim]Anchor: before '{subject_q}' → [0, {t_first:.1f}s)[/dim]")
                    elif anchor_type == "after":
                        anchor_start = t_last if t_last else t_first
                        console.print(f"    [dim]Anchor: after '{subject_q}' → ({anchor_start:.1f}s, end][/dim]")
                else:
                    console.print(f"    [yellow]Anchor subject '{subject_q}' not found — ignoring anchor[/yellow]")

        # Short-circuit explicit timestamp queries to avoid vector search hallucinations
        import re
        ts_pattern = r'(?:from\s+)?(?:(\d+):)?(\d+)\s*(?:s|sec|second|seconds|m|min|minute|minutes)?\s*(?:to|-|and)\s*(?:(\d+):)?(\d+)\s*(?:s|sec|second|seconds|m|min|minute|minutes|nd|th|st|rd)?'
        match = re.search(ts_pattern, query_lower)
        if match and any(kw in query_lower for kw in ["second", "minute", "s", "m", ":", "-", "to", "and", "from", "nd", "th"]):
            # Extract just the timestamp results from structured search
            fused_all = _structured_search(query_lower, modalities, db)
            fused = [s for s in fused_all if s.evidence and any(e.source == "explicit_timestamp" for e in s.evidence)]
            if fused:
                console.print(f"    [dim]Short-circuited semantic search for explicit timestamp query: '{query}'[/dim]")
                results.append(RetrievalResult(operation_index=i, segments=fused))
                continue

        # Short-circuit low-level audio queries to avoid semantic scene matching
        is_silence = any(kw in query_lower for kw in ["silence", "pause", "dead time", "dead air", "gap"])
        is_filler = any(kw in query_lower for kw in ["filler", "um", "uh", "hmm"])
        
        if (is_silence or is_filler) and not any(kw in query_lower for kw in ["scene", "shot", "topic", "part"]):
            fused = []
            if is_silence:
                fused.extend(_structured_search("silence", modalities, db))
            if is_filler:
                fused.extend(_structured_search("filler", modalities, db))
                
            console.print(f"    [dim]Short-circuited semantic search for VAD query: '{query}' ({len(fused)} found)[/dim]")
            results.append(RetrievalResult(operation_index=i, segments=fused))
            continue

        fused = _resolve_compound_intersection(query, modalities, project_dir, db)
        channel_results: dict[str, list[Segment]] = {}

        if not fused:
            
            expanded_queries = _expand_query(query)
            for eq in expanded_queries:
                structured = _structured_search(eq, modalities, db)
                keyword = _keyword_search(eq, project_dir, db)
                vector = _vector_search(eq, project_dir, db)
                metadata = _metadata_search(eq, db)

                channel_results = {
                    "structured": structured,
                    "keyword": keyword,
                    "vector": vector,
                    "metadata": metadata,
                }
                fused.extend(_reciprocal_rank_fusion(channel_results, CFG.retrieval.fusion_weights, threshold=threshold))

        
        try:
            import math
            re_ranker = _get_cross_encoder()
            passages = [(query, " ".join(e.detail for e in seg.evidence)) for seg in fused]
            if passages:
                scores = re_ranker.predict(passages)
                for idx, score in enumerate(scores):
                    prob = 1.0 / (1.0 + math.exp(-float(score)))
                    is_metadata = any(e.source not in ("vector", "keyword") for e in fused[idx].evidence)
                    if is_metadata:
                        
                        fused[idx].score = max(fused[idx].score, prob)
                    else:
                        fused[idx].score = prob
        except Exception as e:
            print(f"CE DEBUG ERROR: {e}")

        
        if channel_results:
            
            fused = [seg for seg in fused if seg.score >= threshold]
            
            for seg in fused:
                requires_verif = any(str(sid) in needs_verif_ids for sid in seg.scene_ids)
                if not requires_verif:
                    continue
                
                active_channels = 0
                for c_name, c_res in channel_results.items():
                    if any(r.start <= seg.end and r.end >= seg.start for r in c_res):
                        active_channels += 1
                if active_channels < 2:
                    seg.needs_confirmation = True

        fused = _merge_adjacent(fused)

        # §3.1: Apply anchor time-range filter
        if anchor_start > 0.0 or anchor_end < float('inf'):
            fused = [s for s in fused if s.start >= anchor_start and s.end <= anchor_end]

        if not fused:
            console.print(f"    [yellow]No matches for \"{query}\"[/yellow]")
            
            target_nouns = [w for w in query.split() if w[0].isupper() or w.lower() in ("pricing", "sponsor")]
            is_confidently_absent = len(target_nouns) > 0

            results.append(RetrievalResult(
                operation_index=i,
                segments=[],
                no_match=True,
                suggestions=["No matches found. Prompt items may be absent."] if is_confidently_absent else None,
            ))
        else:
            console.print(f"    Found {len(fused)} segments (best score: {fused[0].score:.2f})")
            results.append(RetrievalResult(
                operation_index=i,
                segments=fused,
            ))

    total_segments = sum(len(res.segments) for res in results)
    if total_segments == 0:
        console.print("    [yellow]⚠ Retrieval yielded 0 segments. Active Fallback: LLM Query Expansion Loop.[/yellow]")
        from trim_engine.llm import call_structured
        from pydantic import BaseModel
        
        class ExpandedQuery(BaseModel):
            terms: list[str]
            
        ops_text = " ".join([f"{op.target.query} ({','.join(op.target.modality)})" for op in intent.operations])
        original_query = intent.operations[0].target.query if intent.operations else "unknown"
        expansion_prompt = f"The user asked to edit a video targeting '{original_query}'. We searched the video transcript and metadata for '{ops_text}' but found exactly 0 matches. Provide 5 alternative, creative synonyms, phrases, or visual descriptions that might be present in a raw video transcript or visual scene to help us find this moment. You must output valid JSON containing exactly one key named 'terms' which maps to a list of strings."
        
        try:
            expansion = call_structured(
                prompt_name="query_expansion", 
                user_content=[{"type": "text", "text": expansion_prompt}],
                schema=ExpandedQuery,
                effort="low",
                db=db
            )
            expanded_terms = expansion.terms
            console.print(f"    [dim]Expanded query terms: {expanded_terms}[/dim]")
            
            fallback_segments = []
            if expanded_terms:
                for term in expanded_terms:
                    fallback_segments.extend(_keyword_search(term, project_dir, db))
                    fallback_segments.extend(_vector_search(term, project_dir, db))
                    
            fused = _reciprocal_rank_fusion({"fallback": fallback_segments}, weights={"fallback": 1.0}, threshold=0.1)[:3]
            
            if fused:
                console.print(f"    [green]Query expansion successful, found {len(fused)} segments![/green]")
                if results:
                    results[0].segments = fused
                    results[0].no_match = False
                else:
                    results.append(RetrievalResult(operation_index=0, segments=fused, no_match=False))
            else:
                console.print("    [red]⚠ Query expansion also yielded 0 segments.[/red]")
        except Exception as e:
            console.print(f"    [red]⚠ Query expansion failed: {e}[/red]")
    return results

def answer_question(question: str, db: ProjectDB, project_dir: Path) -> str:
    structured = _structured_search(question, ["visual", "speech", "person", "object", "location"], db)
    keyword = _keyword_search(question, project_dir, db)
    vector = _vector_search(question, project_dir, db)
    metadata = _metadata_search(question, db)

    all_segments = structured + keyword + vector + metadata
    all_segments = sorted(all_segments, key=lambda s: s.score, reverse=True)[:10]

    if not all_segments:
        return "I couldn't find information about that in this video."


    context_lines = []
    for s in all_segments:
        context_lines.append(f"[{s.start:.1f}s-{s.end:.1f}s]: " + "; ".join([e.detail for e in s.evidence]))
    
    context_text = "\n".join(context_lines)
    prompt = (
        f"The user asked: '{question}'.\n\n"
        f"Here are some relevant clips retrieved from the video:\n{context_text}\n\n"
        f"Please synthesize a concise, helpful answer. If the context doesn't fully answer the question, state what you know. Cite the timestamps (e.g., 'At 15.5s, ...')."
    )
    
    try:
        from rich.console import Console
        from pydantic import BaseModel
        
        class AnswerResponse(BaseModel):
            answer: str
            
        Console().print("  [dim]Synthesizing answer via LLM...[/dim]")
        from trim_engine.llm import call_structured
        res = call_structured("kb_answer", [{"type": "text", "text": prompt}], schema=AnswerResponse, db=db, effort="low")
        return res.answer
    except Exception as e:
        # Fallback to raw output if LLM fails
        Console().print(f"  [red]LLM error: {e}[/red]")
        lines = [f"• [{seg.start:.1f}s–{seg.end:.1f}s] {ev.detail}" for seg in all_segments for ev in seg.evidence]
        return "\n".join(lines[:15])
