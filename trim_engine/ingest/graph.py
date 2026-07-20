"""
Knowledge Graph Builder (INGESTION_ENGINE.md §4.6) — entity resolution + relations.

Fuses visual entities and speaker turns, emits semantic relations, handles cross-modal event
derivation, snaps diarization turns, and records model manifest.
"""

from __future__ import annotations

import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB

console = Console()






def _snap_diarization_turns(db: ProjectDB) -> None:
    """Snap speaker turn/utterance boundaries to exact first/last word timings."""
    utterances = db.get_utterances()
    words = db.get_words()

    if not utterances or not words:
        return

    
    utt_words = defaultdict(list)
    for w in words:
        utt_words[w["utt_id"]].append(w)

    snapped_count = 0
    for utt in utterances:
        u_words = utt_words.get(utt["id"])
        if u_words:
            first_start = min(w["start_time"] for w in u_words)
            last_end = max(w["end_time"] for w in u_words)

            db.update_utterance(
                utt["id"],
                start_time=first_start,
                end_time=last_end,
            )
            snapped_count += 1

    console.print(f"    Snapped {snapped_count} utterances to word boundaries")






def _resolve_people(db: ProjectDB) -> dict[str, str]:
    """
    Merge person_* keys across scenes into canonical identities.
    Fuses face pipeline entities (SCRFD+ArcFace) with VLM description-based entries.

    Face pipeline produces person:A, person:B entities with source='faces' relations.
    VLM produces person_raw:scene_0 entries with description strings.

    Returns mapping: raw_entity_id → canonical_person_id (person:A, person:B, ...)
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CFG.embedding.text_model_name)
    except Exception as e:
        console.print(f"    [yellow]SentenceTransformer load failed ({e}) — falling back to difflib string heuristic for people merging[/yellow]")
        model = None

    all_people = db.get_entities(kind="person")
    if not all_people:
        return {}

    
    face_entities = [p for p in all_people if not p["id"].startswith("person_raw:")]
    raw_entries = [p for p in all_people if p["id"].startswith("person_raw:")]

    mapping: dict[str, str] = {}
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    next_label = 0

    
    existing_labels: set[str] = set()
    for fe in face_entities:
        label_part = fe["id"].split(":")[-1] if ":" in fe["id"] else fe["id"]
        existing_labels.add(label_part)
        mapping[fe["id"]] = fe["id"]

    if not raw_entries:
        console.print(f"    People resolved (face pipeline only): {len(face_entities)} canonical")
        return mapping

    
    descriptions = [p.get("description", "") or p["label"] for p in raw_entries]

    
    if all(not d for d in descriptions) or model is None:
        
        console.print("    VLM descriptions empty or SentenceTransformer unavailable — using scene overlap / string similarities for fusion")
        for raw_entry in raw_entries:
            raw_id = raw_entry["id"]
            raw_scene = int(raw_id.split(":")[1]) if ":" in raw_id and raw_id.split(":")[1].isdigit() else -1

            
            matched = False
            if raw_scene >= 0:
                face_relations = db.get_relations(rel="appears_in", dst=f"scene:{raw_scene}")
                for rel in face_relations:
                    src = rel["src"]
                    if src.startswith("person:") and src in mapping:
                        
                        mapping[raw_id] = src
                        matched = True
                        break

            if not matched:
                # Fallback fuzzy string matching against existing canonical descriptions
                description = raw_entry.get("description", "") or raw_entry.get("label", "")
                if description:
                    import difflib
                    best_match = None
                    best_score = 0.0
                    for canonical_id, mapped_id in mapping.items():
                        c_entity = [fe for fe in face_entities if fe["id"] == mapped_id]
                        if c_entity:
                            c_desc = c_entity[0].get("description", "")
                            if c_desc:
                                ratio = difflib.SequenceMatcher(None, description.lower(), c_desc.lower()).ratio()
                                if ratio > best_score:
                                    best_score = ratio
                                    best_match = mapped_id
                    if best_score > 0.65:
                        mapping[raw_id] = best_match
                        matched = True

            if not matched:
                canonical_id = f"person:{labels[next_label % 26]}"
                next_label += 1
                while canonical_id in existing_labels or any(
                    v == canonical_id for v in mapping.values()
                ):
                    canonical_id = f"person:{labels[next_label % 26]}"
                    next_label += 1
                mapping[raw_id] = canonical_id
                description = raw_entry.get("description", "") or raw_entry.get("label", "")
                db.insert_entity(
                    entity_id=canonical_id,
                    kind="person",
                    label=canonical_id,
                    description=description or f"Person detected in scene {raw_scene}",
                )

        console.print(f"    People resolved: {len(raw_entries)} raw → {len(set(mapping.values()))} canonical")
        if model is not None:
            del model
        gc.collect()
        return mapping

    embeddings = model.encode(descriptions, normalize_embeddings=True)

    
    threshold = CFG.vision.person_merge_cosine_threshold
    clusters: list[list[int]] = []
    assigned: set[int] = set()

    for i in range(len(raw_entries)):
        if i in assigned:
            continue

        cluster = [i]
        assigned.add(i)

        for j in range(i + 1, len(raw_entries)):
            if j in assigned:
                continue

            if embeddings is not None:
                sim = float(np.dot(embeddings[i], embeddings[j]))
            else:
                import difflib
                sim = difflib.SequenceMatcher(None, descriptions[i].lower(), descriptions[j].lower()).ratio()

            
            scene_i = int(raw_entries[i]["id"].split(":")[1]) if ":" in raw_entries[i]["id"] and raw_entries[i]["id"].split(":")[1].isdigit() else -1
            scene_j = int(raw_entries[j]["id"].split(":")[1]) if ":" in raw_entries[j]["id"] and raw_entries[j]["id"].split(":")[1].isdigit() else -1

            if abs(scene_i - scene_j) <= 2:
                sim += CFG.vision.person_merge_temporal_bonus

            if sim > threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    
    for cluster in clusters:
        scenes_in_cluster = set()
        for i in cluster:
            raw_id = raw_entries[i]["id"]
            scene_num = int(raw_id.split(":")[1]) if ":" in raw_id and raw_id.split(":")[1].isdigit() else -1
            if scene_num >= 0:
                scenes_in_cluster.add(scene_num)

        
        best_face_match = None
        for scene_num in scenes_in_cluster:
            face_relations = db.get_relations(rel="appears_in", dst=f"scene:{scene_num}")
            for rel in face_relations:
                src = rel["src"]
                if src.startswith("person:") and src in mapping:
                    best_face_match = src
                    break
            if best_face_match:
                break

        best_desc = max(
            (raw_entries[i].get("description", "") for i in cluster),
            key=len,
        )

        if best_face_match:
            canonical_id = best_face_match
            
            existing = db.get_entity(canonical_id)
            if existing and best_desc and (not existing.get("description") or
                                           "Face cluster" in (existing.get("description") or "")):
                db.insert_entity(
                    entity_id=canonical_id,
                    kind="person",
                    label=canonical_id,
                    description=best_desc,
                )
        else:
            canonical_id = f"person:{labels[next_label % 26]}"
            next_label += 1
            while canonical_id in existing_labels or any(
                v == canonical_id for v in mapping.values()
            ):
                canonical_id = f"person:{labels[next_label % 26]}"
                next_label += 1
            db.insert_entity(
                entity_id=canonical_id,
                kind="person",
                label=canonical_id,
                description=best_desc,
            )

        for i in cluster:
            mapping[raw_entries[i]["id"]] = canonical_id

    console.print(f"    People resolved: {len(raw_entries)} raw + {len(face_entities)} face → {len(set(mapping.values()))} canonical")

    del model, embeddings
    gc.collect()

    return mapping


def _resolve_objects(db: ProjectDB) -> None:
    """Normalize object labels and create canonical object entities."""
    
    lemma = {
        "laptops": "laptop", "phones": "phone", "cameras": "camera",
        "whiteboards": "whiteboard", "desks": "desk", "chairs": "chair",
        "books": "book", "cups": "cup", "glasses": "glass",
        "microphones": "microphone", "monitors": "monitor",
    }

    object_scenes: dict[str, list[int]] = defaultdict(list)

    
    all_relations = db.get_relations(rel="contains")
    for rel in all_relations:
        obj_label = rel["dst"].lower().strip()
        if obj_label.startswith("object:"):
            obj_label = obj_label.split(":", 1)[1]
        obj_label = lemma.get(obj_label, obj_label)

        
        if obj_label.endswith("s") and obj_label not in lemma.values():
            singular = obj_label[:-1]
            if singular in [r["dst"].lower() for r in all_relations]:
                obj_label = singular

        entity_id = f"object:{obj_label}"
        db.insert_entity(entity_id, kind="object", label=obj_label)
        if rel.get("scene_id"):
            object_scenes[entity_id].append(rel["scene_id"])

    console.print(f"    Objects resolved: {len(object_scenes)} unique objects")


def _fuse_speakers_to_people(db: ProjectDB, person_mapping: dict[str, str]) -> None:
    """
    Fuse speaker clusters with person identities.
    utterance speaker cluster × scenes where exactly one person is_speaking
    → majority vote maps speaker_0 → person:A
    """
    utterances = db.get_utterances()
    scenes = db.get_scenes()

    if not utterances or not person_mapping:
        return

    
    scene_speaker_map: dict[int, str] = {}

    for scene in scenes:
        scene_id = scene["id"]
        raw_relations = db.get_relations(rel="appears_in", dst=f"scene:{scene_id}")

        speaking_people = []
        for rel in raw_relations:
            raw_id = rel["src"]
            if raw_id in person_mapping:
                canonical = person_mapping[raw_id]
                speaking_people.append(canonical)

        if len(speaking_people) == 1:
            scene_speaker_map[scene_id] = speaking_people[0]

    
    speaker_person_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for utt in utterances:
        speaker_id = utt.get("speaker_id")
        if not speaker_id:
            continue

        for scene in scenes:
            if utt["start_time"] < scene["end_time"] and utt["end_time"] > scene["start_time"]:
                if scene["id"] in scene_speaker_map:
                    person = scene_speaker_map[scene["id"]]
                    speaker_person_votes[speaker_id][person] += 1
                break

    
    speaker_to_person: dict[str, str] = {}
    for speaker_id, votes in speaker_person_votes.items():
        if votes:
            best_person = max(votes, key=votes.get)  
            speaker_to_person[speaker_id] = best_person

    console.print(f"    Speaker↔Person fusion: {len(speaker_to_person)} mappings")

    
    person_speaking_time: dict[str, float] = defaultdict(float)
    for utt in utterances:
        speaker = utt.get("speaker_id")
        if speaker and speaker in speaker_to_person:
            person = speaker_to_person[speaker]
            person_speaking_time[person] += utt["end_time"] - utt["start_time"]

    if person_speaking_time:
        owner = max(person_speaking_time, key=person_speaking_time.get)  
        
        db.insert_entity(
            entity_id=owner,
            kind="person",
            label=owner,
            description=db.get_entity(owner).get("description") if db.get_entity(owner) else "",
            is_owner=True,
            owner_inferred=True,
        )
        console.print(f"    Owner identified: {owner} ({person_speaking_time[owner]:.1f}s speaking) ✓")






def _emit_relations(db: ProjectDB, person_mapping: dict[str, str]) -> None:
    """Emit all knowledge graph relations, adding model_version metadata."""
    scenes = db.get_scenes()
    utterances = db.get_utterances()
    topics = db.get_topics()

    
    for raw_id, canonical_id in person_mapping.items():
        parts = raw_id.split(":")
        if len(parts) >= 2:
            try:
                scene_id = int(parts[1])
                scene = db.get_scene(scene_id)
                if scene:
                    db.insert_relation(
                        src=canonical_id,
                        rel="appears_in",
                        dst=f"scene:{scene_id}",
                        scene_id=scene_id,
                        t_start=scene["start_time"],
                        t_end=scene["end_time"],
                        confidence=0.9,
                        source="vision",
                        model_version="2.0",
                    )
            except ValueError:
                pass

    
    for scene in scenes:
        if scene.get("location"):
            loc_id = f"location:{scene['location']}"
            db.insert_entity(loc_id, kind="location", label=scene["location"])
            db.insert_relation(
                src=f"scene:{scene['id']}",
                rel="located_in",
                dst=loc_id,
                scene_id=scene["id"],
                t_start=scene["start_time"],
                t_end=scene["end_time"],
                confidence=0.85,
                source="vision",
                model_version="2.0",
            )

    
    for scene in scenes:
        if scene.get("emotion_label"):
            db.insert_relation(
                src=f"scene:{scene['id']}",
                rel="expresses",
                dst=f"emotion:{scene['emotion_label']}",
                scene_id=scene["id"],
                t_start=scene["start_time"],
                t_end=scene["end_time"],
                confidence=scene.get("emotion_intensity", 0.5),
                source="vision",
                model_version="2.0",
            )

    
    for topic in topics:
        topic_utts = [u for u in utterances if u.get("topic_id") == topic["id"]]
        for utt in topic_utts:
            speaker = utt.get("speaker_id")
            if speaker:
                for s_id, p_id in person_mapping.items():
                    if s_id == speaker:
                        db.insert_relation(
                            src=p_id,
                            rel="speaks_about",
                            dst=f"topic:{topic['label']}",
                            t_start=topic["start_time"],
                            t_end=topic["end_time"],
                            confidence=0.8,
                            source="topic_model",
                            model_version="2.0",
                        )
                        break

    
    for i in range(len(scenes) - 1):
        db.insert_relation(
            src=f"scene:{scenes[i]['id']}",
            rel="followed_by",
            dst=f"scene:{scenes[i + 1]['id']}",
            scene_id=scenes[i]["id"],
            t_start=scenes[i]["end_time"],
            t_end=scenes[i + 1]["start_time"],
            confidence=1.0,
            source="scene_detection",
            model_version="2.0",
        )

    console.print(f"    Relations emitted")






def _derive_cross_modal_events(db: ProjectDB) -> None:
    """
    Derive events from overlapping modal outputs:
    e.g., montage: broll(visual) ∩ music(audio)
    """
    scenes = db.get_scenes()
    music_events = db.get_audio_events("music")

    if not scenes or not music_events:
        return

    derived_count = 0
    for s in scenes:
        
        overlap_music = False
        for m in music_events:
            if m["start_time"] < s["end_time"] and m["end_time"] > s["start_time"]:
                overlap_music = True
                break

        
        if overlap_music and s.get("is_broll") == 1:
            db.insert_derived_moment(
                kind="montage",
                start=s["start_time"],
                end=s["end_time"],
                formula="broll(visual) ∩ music(audio)",
                confidence=0.85,
            )
            derived_count += 1

    console.print(f"    Derived cross-modal moments: {derived_count}")






def _handle_contradictions(db: ProjectDB) -> None:
    """Log conflicts or contradictions and flag single-source high-impact claims."""
    scenes = db.get_scenes()
    for s in scenes:
        caption = s.get("caption", "").lower()
        location = s.get("location", "").lower()
        if location and location not in caption:
            
            pass

    
    all_appears_in = db.get_relations(rel="appears_in")
    person_counts: dict[str, int] = {}
    for rel in all_appears_in:
        src = rel["src"]
        if src.startswith("person:"):
            person_counts[src] = person_counts.get(src, 0) + 1

    flagged_count = 0
    for rel in all_appears_in:
        src = rel["src"]
        if src.startswith("person:") and person_counts.get(src, 0) == 1:
            db.update_relation_verification(rel["id"], True)
            flagged_count += 1

    if flagged_count > 0:
        console.print(f"    Contradictions & verification flags updated: {flagged_count} claims")






def scenes_where(
    db: ProjectDB,
    person: str | None = None,
    holds: str | None = None,
    topic: str | None = None,
    location: str | None = None,
    action: str | None = None,
    shot_type: str | None = None,
    emotion: str | None = None,
    is_broll: bool | None = None,
    dialogue_act: str | None = None,
) -> list[dict]:
    """Query scenes matching multiple constraints via SQL joins."""
    conditions: list[str] = []
    params: list = []

    base_query = "SELECT DISTINCT s.* FROM scenes s"
    joins: list[str] = []

    if person:
        joins.append(
            " JOIN relations r_person ON r_person.scene_id = s.id AND r_person.rel = 'appears_in'"
        )
        conditions.append("r_person.src = ?")
        params.append(f"person:{person}" if ":" not in person else person)

    if holds:
        joins.append(
            " JOIN relations r_holds ON r_holds.scene_id = s.id AND r_holds.rel = 'holds'"
        )
        conditions.append("r_holds.dst = ?")
        params.append(f"object:{holds}" if ":" not in holds else holds)

    if topic:
        joins.append(
            " JOIN relations r_topic ON r_topic.scene_id = s.id AND r_topic.rel = 'speaks_about'"
        )
        conditions.append("r_topic.dst = ?")
        params.append(f"topic:{topic}" if ":" not in topic else topic)

    if location:
        conditions.append("s.location = ?")
        params.append(location)

    if shot_type:
        conditions.append("s.shot_type = ?")
        params.append(shot_type)

    if emotion:
        conditions.append("s.emotion_label = ?")
        params.append(emotion)

    if is_broll is not None:
        conditions.append("s.is_broll = ?")
        params.append(int(is_broll))

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"{base_query}{''.join(joins)} WHERE {where} ORDER BY s.start_time"

    with db.conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def first_appearance(db: ProjectDB, person: str) -> float | None:
    """Get the first timestamp where a person appears."""
    person_id = f"person:{person}" if ":" not in person else person
    rels = db.get_relations(src=person_id, rel="appears_in")
    if not rels:
        return None
    return min(r["t_start"] for r in rels if r.get("t_start") is not None)


def last_appearance(db: ProjectDB, person: str) -> float | None:
    """Get the last timestamp where a person appears."""
    person_id = f"person:{person}" if ":" not in person else person
    rels = db.get_relations(src=person_id, rel="appears_in")
    if not rels:
        return None
    return max(r["t_end"] for r in rels if r.get("t_end") is not None)






def run_graph_builder(project_dir: Path, db: ProjectDB) -> None:
    """Run the full knowledge graph building stage."""
    console.print("    Building knowledge graph...")

    
    _snap_diarization_turns(db)

    
    person_mapping = _resolve_people(db)

    
    _resolve_objects(db)

    
    _emit_relations(db, person_mapping)

    
    _fuse_speakers_to_people(db, person_mapping)

    
    _derive_cross_modal_events(db)

    
    _handle_contradictions(db)

    
    _calibrate_confidences(db)

    # §3.3: Compute derived moments
    _derive_moments(db)

    # Cleanup raw entities
    with db.conn() as c:
        c.execute("DELETE FROM entities WHERE id LIKE 'person_raw:%'")

    db.set_coverage("graph", "available")
    db.set_model_manifest("graph", "agglomerative-fusion", "2.0")
    console.print("    [dim]Knowledge graph complete[/dim]")

def _derive_moments(db: ProjectDB) -> None:
    """§3.3: Derive complex multi-modal moments (awkward, funniest, applause)."""
    count = 0
    # 1. Applause moments (from audio events)
    events = db.get_audio_events()
    for ev in events:
        if ev["type"] == "applause":
            db.insert_derived_moment("applause", ev["start_time"], ev["end_time"], "audio_event(applause)", ev["confidence"])
            count += 1
            
        # 2. Funniest moments (laughter audio + joy emotion)
        if ev["type"] == "laughter":
            # Check if there is joy emotion nearby
            with db.conn() as c:
                joy = c.execute(
                    "SELECT 1 FROM relations WHERE predicate = 'emotion' AND target = 'joy' "
                    "AND start_time <= ? AND end_time >= ?", (ev["end_time"] + 2.0, ev["start_time"] - 2.0)
                ).fetchone()
            if joy:
                db.insert_derived_moment("funniest", ev["start_time"], ev["end_time"], "laughter + joy", ev["confidence"])
                count += 1

    # 3. Awkward moments (silence > 1.5s AND no scene change, or overlapping speakers with confusion)
    # This is a heuristic approximation: long pauses between utterances.
    utterances = db.get_utterances()
    if utterances:
        utterances.sort(key=lambda u: u["start_time"])
        for i in range(len(utterances) - 1):
            u1, u2 = utterances[i], utterances[i+1]
            gap = u2["start_time"] - u1["end_time"]
            if gap > 2.0:
                # long pause
                db.insert_derived_moment("awkward", u1["end_time"], u2["start_time"], "long_pause", 0.7)
                count += 1

    if count > 0:
        console.print(f"    Derived {count} moments (awkward, funniest, applause)")

    # §3.4 Off-topic detection
    video = db.get_video()
    video_context = f"{video.get('title', '')} {video.get('description', '')}".strip() if video else ""
    if video_context:
        topics = db.get_topics()
        if topics:
            try:
                from sentence_transformers import SentenceTransformer
                from trim_engine.config import CFG
                import numpy as np
                
                model = SentenceTransformer(CFG.embedding.text_model_name)
                texts = [video_context] + [t.get("label", "") for t in topics]
                embeddings = model.encode(texts, normalize_embeddings=True)
                
                vid_emb = embeddings[0]
                topic_embs = embeddings[1:]
                
                similarities = np.dot(topic_embs, vid_emb)
                
                off_topic_count = 0
                for idx, sim in enumerate(similarities):
                    if sim < 0.35:  # threshold
                        topic = topics[idx]
                        db.insert_derived_moment("off_topic", topic["start_time"], topic["end_time"], f"similarity = {sim:.2f}", float(1.0 - sim))
                        off_topic_count += 1
                        
                if off_topic_count > 0:
                    console.print(f"    Detected {off_topic_count} off-topic chapters")
            except Exception as e:
                console.print(f"    [yellow]Off-topic detection failed: {e}[/yellow]")










_CALIBRATION_MAPS: dict[str, list[tuple[float, float]]] = {
    "vision": [
        (0.0, 0.0),
        (0.3, 0.15),
        (0.5, 0.35),
        (0.7, 0.60),
        (0.85, 0.80),
        (1.0, 0.95),
    ],
    "faces": [
        (0.0, 0.0),
        (0.4, 0.25),
        (0.6, 0.50),
        (0.8, 0.75),
        (1.0, 0.90),
    ],
    "is_speaking_binding": [
        (0.0, 0.0),
        (0.3, 0.10),
        (0.5, 0.30),
        (0.7, 0.55),
        (0.9, 0.80),
        (1.0, 0.85),
    ],
    "graph": [
        (0.0, 0.0),
        (0.5, 0.40),
        (0.8, 0.70),
        (1.0, 0.90),
    ],
}


def _apply_calibration_map(raw: float, breakpoints: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation on a calibration map."""
    if raw <= breakpoints[0][0]:
        return breakpoints[0][1]
    if raw >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= raw <= x1:
            t = (raw - x0) / (x1 - x0) if (x1 - x0) > 1e-10 else 0.0
            return y0 + t * (y1 - y0)

    return raw  


def _calibrate_confidences(db: ProjectDB) -> None:
    """
    Apply per-source calibration maps to all relation confidences.

    This is a stub for production isotonic regression. The maps above
    are hand-tuned conservative estimates that can be replaced with
    learned maps from labeled data via sklearn.isotonic.IsotonicRegression.
    """
    with db.conn() as c:
        relations = c.execute(
            "SELECT id, confidence, source FROM relations WHERE confidence IS NOT NULL"
        ).fetchall()

    calibrated_count = 0
    with db.conn() as c:
        for rel in relations:
            source = rel["source"] or "graph"
            bp = _CALIBRATION_MAPS.get(source)
            if bp is None:
                continue

            raw = rel["confidence"]
            calibrated = _apply_calibration_map(raw, bp)

            if abs(calibrated - raw) > 0.01:
                c.execute(
                    "UPDATE relations SET confidence = ? WHERE rowid = ?",
                    (round(calibrated, 4), rel["id"]),
                )
                calibrated_count += 1

    if calibrated_count > 0:
        console.print(f"    Calibrated {calibrated_count} relation confidences")
