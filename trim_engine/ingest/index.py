"""
Semantic Index Engine (INGESTION_ENGINE.md §4.7) — FAISS + BM25 + metadata indexes.
"""

from __future__ import annotations

import gc
import os
import pickle
from pathlib import Path

import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

console = Console()



def _build_clip_scene_index_resnet_fallback(project_dir: Path, db: ProjectDB) -> None:
    """Active Alternative Path: Use Torchvision ResNet-50 instead of OpenCLIP."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import gc
    import os
    import pickle
    import faiss

    model_name = "ResNet-50"
    model_version = "torchvision/resnet50"
    dim = 2048

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    scenes = db.get_scenes()
    if not scenes:
        return

    scene_embeddings: list[np.ndarray] = []
    frame_embeddings: list[np.ndarray] = []
    frame_lut_mapping: list[dict] = []
    
    vector_id = 0

    for scene in scenes:
        keyframes = db.get_keyframes(scene["id"])
        kf_paths = [Path(kf["path"]) for kf in keyframes if Path(kf["path"]).exists()]

        if not kf_paths:
            scene_embeddings.append(np.zeros(dim, dtype=np.float32))
            db.insert_scene_vector(vector_id, scene["id"], "clip", model_version=model_version)
            vector_id += 1
            continue

        kf_embeddings = []
        for kf_idx, kf_path in enumerate(kf_paths):
            try:
                image = preprocess(Image.open(kf_path).convert('RGB')).unsqueeze(0)
                with torch.no_grad():
                    emb = model(image).flatten()
                    
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    emb_np = emb.numpy()
                    kf_embeddings.append(emb_np)
                    
                    frame_embeddings.append(emb_np)
                    frame_lut_mapping.append({
                        "frame_idx": len(frame_embeddings) - 1,
                        "scene_id": scene["id"],
                        "kf_path": str(kf_path)
                    })
            except Exception:
                continue

        if kf_embeddings:
            mean_emb = np.mean(kf_embeddings, axis=0).astype(np.float32)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            scene_embeddings.append(mean_emb)
        else:
            scene_embeddings.append(np.zeros(dim, dtype=np.float32))

        db.insert_scene_vector(vector_id, scene["id"], "clip", model_version=model_version)
        vector_id += 1

    faiss_dir = project_dir / "faiss"
    faiss_dir.mkdir(exist_ok=True)

    if scene_embeddings:
        index_scene = faiss.IndexFlatIP(dim)
        matrix_scene = np.array(scene_embeddings, dtype=np.float32)
        index_scene.add(matrix_scene)
        
        tmp_idx = faiss_dir / "scene_clip.index.tmp"
        faiss.write_index(index_scene, str(tmp_idx))
        os.replace(tmp_idx, faiss_dir / "scene_clip.index")

    if frame_embeddings:
        index_frame = faiss.IndexFlatIP(dim)
        matrix_frame = np.array(frame_embeddings, dtype=np.float32)
        index_frame.add(matrix_frame)
        
        tmp_idx = faiss_dir / "frame_clip.index.tmp"
        faiss.write_index(index_frame, str(tmp_idx))
        os.replace(tmp_idx, faiss_dir / "frame_clip.index")
        
        tmp_map = faiss_dir / "frame_lut_map.pkl.tmp"
        with open(tmp_map, "wb") as f:
            pickle.dump(frame_lut_mapping, f)
        os.replace(tmp_map, faiss_dir / "frame_lut_map.pkl")

    console.print(f"    Image index built (ResNet fallback): {len(scene_embeddings)} scene vectors, {len(frame_embeddings)} frame vectors, dim={dim}")
    db.set_model_manifest("clip_embedding", model_name, "imagenet")
    del model
    gc.collect()

def _build_clip_scene_index(project_dir: Path, db: ProjectDB) -> None:
    """Build SigLIP/CLIP image embedding index (both per-frame and mean-pooled scene index)."""
    console.print("    [dim]Building image scene index (SigLIP / CLIP)...[/dim]")

    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception as e:
        console.print(f"    [yellow]⚠ CLIP dependencies failed to load ({e}). Active Fallback: Torchvision ResNet-50.[/yellow]")
        return _build_clip_scene_index_resnet_fallback(project_dir, db)

    scenes = db.get_scenes()
    if not scenes:
        return

    
    model = None
    preprocess = None
    model_name = "ViT-B-16-SigLIP"
    pretrained = "webli"
    dim = 768  

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        console.print(f"    Loaded model: {model_name} ({pretrained})")
    except Exception as e:
        console.print(f"    [yellow]SigLIP load failed ({e}), falling back to CLIP ViT-B-32[/yellow]")
        model_name = "ViT-B-32"
        pretrained = "openai"
        dim = 512
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

    model.eval()

    scene_embeddings: list[np.ndarray] = []
    frame_embeddings: list[np.ndarray] = []
    frame_lut_mapping: list[dict] = []  
    
    vector_id = 0
    model_version = f"{model_name}/{pretrained}"

    for scene in scenes:
        keyframes = db.get_keyframes(scene["id"])
        kf_paths = [Path(kf["path"]) for kf in keyframes if Path(kf["path"]).exists()]

        if not kf_paths:
            
            scene_embeddings.append(np.zeros(dim, dtype=np.float32))
            db.insert_scene_vector(vector_id, scene["id"], "clip", model_version=model_version)
            vector_id += 1
            continue

        kf_embeddings = []
        for kf_idx, kf_path in enumerate(kf_paths):
            try:
                image = preprocess(Image.open(kf_path)).unsqueeze(0)
                with torch.no_grad():
                    emb = model.encode_image(image)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    emb_np = emb.squeeze().numpy()
                    kf_embeddings.append(emb_np)
                    
                    
                    frame_embeddings.append(emb_np)
                    frame_lut_mapping.append({
                        "frame_idx": len(frame_embeddings) - 1,
                        "scene_id": scene["id"],
                        "kf_path": str(kf_path)
                    })
            except Exception:
                continue

        if kf_embeddings:
            mean_emb = np.mean(kf_embeddings, axis=0).astype(np.float32)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            scene_embeddings.append(mean_emb)
        else:
            scene_embeddings.append(np.zeros(dim, dtype=np.float32))

        db.insert_scene_vector(vector_id, scene["id"], "clip", model_version=model_version)
        vector_id += 1

    
    import faiss
    faiss_dir = project_dir / "faiss"
    faiss_dir.mkdir(exist_ok=True)

    
    if scene_embeddings:
        index_scene = faiss.IndexFlatIP(dim)
        matrix_scene = np.array(scene_embeddings, dtype=np.float32)
        index_scene.add(matrix_scene)
        
        tmp_idx = faiss_dir / "scene_clip.index.tmp"
        faiss.write_index(index_scene, str(tmp_idx))
        os.replace(tmp_idx, faiss_dir / "scene_clip.index")

    
    if frame_embeddings:
        index_frame = faiss.IndexFlatIP(dim)
        matrix_frame = np.array(frame_embeddings, dtype=np.float32)
        index_frame.add(matrix_frame)
        
        tmp_idx = faiss_dir / "frame_clip.index.tmp"
        faiss.write_index(index_frame, str(tmp_idx))
        os.replace(tmp_idx, faiss_dir / "frame_clip.index")
        
        
        tmp_map = faiss_dir / "frame_lut_map.pkl.tmp"
        with open(tmp_map, "wb") as f:
            pickle.dump(frame_lut_mapping, f)
        os.replace(tmp_map, faiss_dir / "frame_lut_map.pkl")

    console.print(f"    Image index built: {len(scene_embeddings)} scene vectors, {len(frame_embeddings)} frame vectors, dim={dim}")

    db.set_model_manifest("clip_embedding", model_name, pretrained)
    del model
    gc.collect()


def _build_text_scene_index(project_dir: Path, db: ProjectDB) -> None:
    """Build MiniLM text embedding index for semantic scene queries."""
    console.print("    [dim]Building text scene index...[/dim]")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CFG.embedding.text_model_name)
    except Exception as e:
        console.print(f"    [yellow]⚠ SentenceTransformer load failed for text scene index ({e}). Skipping text scene indexing.[/yellow]")
        return
    model_version = CFG.embedding.text_model_name

    
    texts: list[str] = []
    for scene in scenes:
        parts = []
        if scene.get("caption"):
            parts.append(scene["caption"])
        if scene.get("location"):
            parts.append(f"location: {scene['location']}")
        if scene.get("emotion_label"):
            parts.append(f"emotion: {scene['emotion_label']}")
        if scene.get("shot_type"):
            parts.append(f"shot: {scene['shot_type']}")
        texts.append(" | ".join(parts) if parts else "empty scene")

    embeddings = model.encode(texts, normalize_embeddings=True)

    clip_offset = len(scenes)
    for i, scene in enumerate(scenes):
        db.insert_scene_vector(clip_offset + i, scene["id"], "text", model_version=model_version)

    if len(embeddings) > 0:
        import faiss

        dim = CFG.embedding.text_dim
        index = faiss.IndexFlatIP(dim)
        matrix = np.array(embeddings, dtype=np.float32)
        index.add(matrix)

        faiss_dir = project_dir / "faiss"
        tmp_idx = faiss_dir / "scene_text.index.tmp"
        faiss.write_index(index, str(tmp_idx))
        os.replace(tmp_idx, faiss_dir / "scene_text.index")

        console.print(f"    Text scene index: {len(embeddings)} vectors, {dim}d (v: {model_version})")

    del model, embeddings
    gc.collect()


def _build_utterance_index(project_dir: Path, db: ProjectDB) -> None:
    """Build MiniLM embedding index for utterance-level search."""
    console.print("    [dim]Building utterance index...[/dim]")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CFG.embedding.text_model_name)
    except Exception as e:
        console.print(f"    [yellow]⚠ SentenceTransformer load failed for utterance index ({e}). Skipping utterance indexing.[/yellow]")
        return
    model_version = CFG.embedding.text_model_name

    texts = [u["text"] for u in utterances]
    embeddings = model.encode(texts, normalize_embeddings=True)

    for i, utt in enumerate(utterances):
        db.insert_utt_vector(i, utt["id"], model_version=model_version)

    if len(embeddings) > 0:
        import faiss

        dim = CFG.embedding.text_dim
        index = faiss.IndexFlatIP(dim)
        matrix = np.array(embeddings, dtype=np.float32)
        index.add(matrix)

        faiss_dir = project_dir / "faiss"
        tmp_idx = faiss_dir / "utterance.index.tmp"
        faiss.write_index(index, str(tmp_idx))
        os.replace(tmp_idx, faiss_dir / "utterance.index")

        console.print(f"    Utterance index: {len(embeddings)} vectors, {dim}d (v: {model_version})")

    db.set_model_manifest("text_embedding", CFG.embedding.text_model_name, "1.0")
    del model, embeddings
    gc.collect()


def _build_bm25_index(project_dir: Path, db: ProjectDB) -> None:
    """Build BM25 keyword index over transcript + captions + labels."""
    console.print("    [dim]Building BM25 index...[/dim]")

    from rank_bm25 import BM25Okapi

    utterances = db.get_utterances()
    scenes = db.get_scenes()

    corpus: list[list[str]] = []
    doc_refs: list[dict] = []

    
    for utt in utterances:
        tokens = utt["text"].lower().split()
        corpus.append(tokens)
        doc_refs.append({"type": "utterance", "id": utt["id"],
                        "start": utt["start_time"], "end": utt["end_time"]})

    
    for scene in scenes:
        parts = []
        if scene.get("caption"):
            parts.extend(scene["caption"].lower().split())
        if scene.get("location"):
            parts.extend(scene["location"].lower().split())
        if scene.get("emotion_label"):
            parts.append(scene["emotion_label"].lower())
        if scene.get("shot_type"):
            parts.append(scene["shot_type"].lower())

        if parts:
            corpus.append(parts)
            doc_refs.append({"type": "scene", "id": scene["id"],
                            "start": scene["start_time"], "end": scene["end_time"]})

    if corpus:
        bm25 = BM25Okapi(corpus)

        faiss_dir = project_dir / "faiss"
        faiss_dir.mkdir(exist_ok=True)

        with open(faiss_dir / "bm25.pkl.tmp", "wb") as f:
            pickle.dump({"bm25": bm25, "doc_refs": doc_refs, "corpus": corpus}, f)
        os.replace(faiss_dir / "bm25.pkl.tmp", faiss_dir / "bm25.pkl")

        console.print(f"    BM25 index: {len(corpus)} documents")
    else:
        console.print("    [dim]No content for BM25 index[/dim]")


def run_index_builder(project_dir: Path, db: ProjectDB) -> None:
    """Run the full index building stage."""
    console.print("    Building indexes...")

    _build_clip_scene_index(project_dir, db)
    _build_text_scene_index(project_dir, db)
    _build_utterance_index(project_dir, db)
    _build_bm25_index(project_dir, db)

    db.set_coverage("indexes", "available")
    db.set_model_manifest("index", "faiss+bm25", "2.0")
    console.print("    [dim]All indexes built[/dim]")
