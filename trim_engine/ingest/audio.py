"""
Audio Intelligence Engine (INGESTION_ENGINE.md §5)
"""

from __future__ import annotations

import gc
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB


console = Console()






def _run_vad(audio_path: Path, db: ProjectDB) -> list[dict]:
    """Detect speech regions via silero-vad; complement = silences."""
    console.print("    [dim](B1) VAD / silence detection...[/dim]")

    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(str(audio_path), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        threshold=CFG.audio.vad_threshold,
        return_seconds=True,
    )

    video = db.get_video()
    total_duration = video["duration_s"] if video else len(wav) / 16000

    silence_count = 0
    prev_end = 0.0

    for ts in speech_timestamps:
        gap = ts["start"] - prev_end
        if gap >= CFG.audio.min_silence_duration_s:
            db.insert_silence(prev_end, ts["start"])
            silence_count += 1
        prev_end = ts["end"]

    if total_duration - prev_end >= CFG.audio.min_silence_duration_s:
        db.insert_silence(prev_end, total_duration)
        silence_count += 1

    console.print(f"    Speech regions: {len(speech_timestamps)}, Silences: {silence_count}")
    
    # Track C3: Room-tone extraction
    # Find the longest silence to use as room tone
    silences = db.get_silences()
    if silences:
        longest_silence = max(silences, key=lambda s: s["end_time"] - s["start_time"])
        if longest_silence["end_time"] - longest_silence["start_time"] > 0.5:
            roomtone_path = audio_path.parent / "roomtone.wav"
            console.print("    [dim](C3) Extracting room-tone from longest silence...[/dim]")
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(longest_silence["start_time"]),
                "-to", str(longest_silence["end_time"]),
                "-i", str(audio_path),
                "-ac", "1", "-ar", "16000",
                str(roomtone_path)
            ]
            subprocess.run(cmd, capture_output=True)
            
    db.set_coverage("vad", "available")

    del model
    gc.collect()
    return speech_timestamps






class _CTCAligner:
    """
    CTC forced alignment with full-audio batched forward pass.

    Architecture:
      1. Load wav2vec2 model + full audio ONCE
      2. Run ONE forward pass on the full audio, cache [T, vocab] logits
      3. For each utterance, slice pre-computed logits and forced-align

    Fallback ladder:
      1. torchaudio forced_align on sliced full-audio logits (frame-level, ±10ms)
      2. Argmax token walking on sliced logits (good: ±20ms)
      3. Character-ratio interpolation (degraded: ±200ms, snap_tolerance='wide')
    """

    def __init__(self, audio_path: Path):
        self._audio_path = audio_path
        self._model = None
        self._processor = None
        self._waveform = None
        self._full_logits = None
        self._fps = None
        self._sr = 16000
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Lazy-load model, full audio, and run ONE forward pass."""
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import torch
            import torchaudio
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

            model_id = CFG.audio.ctc_alignment_model
            self._processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_id)
            self._model.eval()

            self._waveform, sr = torchaudio.load(str(self._audio_path))
            if sr != self._sr:
                self._waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self._sr)(self._waveform)
            self._waveform = self._waveform.squeeze(0)

            inputs = self._processor(self._waveform, sampling_rate=self._sr, return_tensors="pt")
            with torch.no_grad():
                self._full_logits = self._model(inputs.input_values).logits.squeeze(0)

            audio_duration = len(self._waveform) / self._sr
            self._fps = self._full_logits.shape[0] / max(audio_duration, 1e-6)

            console.print(f"      [dim]CTC aligner: {model_id} "
                         f"({self._full_logits.shape[0]} frames @ {self._fps:.1f} Hz)[/dim]")
            return True
        except Exception as e:
            console.print(f"      [yellow]⚠ CTC aligner failed: {e} — interpolation fallback[/yellow]")
            self._model = None
            return False

    def align_utterance(self, text: str, start_s: float, end_s: float) -> tuple[list[dict], str]:
        words = text.split()
        if not words:
            return [], "normal"

        if not self._ensure_loaded():
            return self._interpolate(words, start_s, end_s), "wide"

        try:
            start_frame = max(0, int(start_s * self._fps))
            end_frame = min(self._full_logits.shape[0], max(start_frame + 1, int(end_s * self._fps)))

            if end_frame - start_frame < 2:
                return self._interpolate(words, start_s, end_s), "wide"

            segment_logits = self._full_logits[start_frame:end_frame]

            aligned = self._try_forced_align(segment_logits, text, start_s, end_s, words)
            if aligned is not None:
                return aligned, "normal"

            aligned = self._try_argmax_alignment(segment_logits, words, start_s, end_s)
            if aligned is not None:
                return aligned, "normal"

            return self._interpolate(words, start_s, end_s), "wide"

        except Exception as e:
            console.print(f"      [yellow]⚠ CTC alignment error: {e} — interpolation[/yellow]")
            return self._interpolate(words, start_s, end_s), "wide"

    def _build_token_ids(self, text: str) -> list[int] | None:
        """Build CTC token ID sequence. Character-level for wav2vec2; falls back to processor."""
        vocab = self._processor.tokenizer.get_vocab()
        clean = re.sub(r"[^A-Z\s]", "", text.strip().upper())
        tokens = []
        for char in clean:
            tid = vocab.get("|") if char == " " else vocab.get(char)
            if tid is None:
                return None
            tokens.append(tid)
        return tokens if tokens else None

    def _try_forced_align(self, logits, text: str, start_s: float, end_s: float, words: list[str]) -> list[dict] | None:
        import torch
        import torchaudio

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        tokens = self._build_token_ids(text)
        if tokens is None:
            encoded = self._processor(text=re.sub(r"[^A-Za-z0-9\s]", "", text).upper(),
                                      return_tensors="pt", add_special_tokens=False)
            tokens = encoded.input_ids[0].tolist()
        if not tokens:
            return None

        token_tensor = torch.tensor([tokens], dtype=torch.int32)
        blank_id = self._processor.tokenizer.pad_token_id
        word_sep_id = self._processor.tokenizer.vocab.get("|")

        try:
            aligned, scores = torchaudio.functional.forced_align(
                log_probs.unsqueeze(0), token_tensor, blank=blank_id,
            )
        except Exception:
            return None

        aligned = aligned.squeeze(0)
        scores = scores.squeeze(0)
        frame_duration = (end_s - start_s) / log_probs.shape[0]

        word_alignments = []
        word_idx = 0
        word_start = None
        word_probs = []

        for frame_idx, tok, score in zip(range(log_probs.shape[0]), aligned.tolist(), scores.tolist()):
            if tok == blank_id:
                continue
            if word_sep_id is not None and tok == word_sep_id:
                if word_start is not None and word_idx < len(words):
                    word_end = start_s + frame_idx * frame_duration
                    word_alignments.append({
                        "word": words[word_idx],
                        "start": word_start,
                        "end": word_end,
                        "prob": float(np.exp(np.mean(word_probs))) if word_probs else 0.5,
                    })
                    word_idx += 1
                    word_probs = []
                word_start = None
            else:
                if word_start is None:
                    word_start = start_s + frame_idx * frame_duration
                word_probs.append(score)

        if word_start is not None and word_idx < len(words):
            word_alignments.append({
                "word": words[word_idx],
                "start": word_start,
                "end": end_s,
                "prob": float(np.exp(np.mean(word_probs))) if word_probs else 0.5,
            })
            word_idx += 1

        if len(word_alignments) >= len(words) * 0.7:
            if len(word_alignments) < len(words):
                last_end = word_alignments[-1]["end"] if word_alignments else start_s
                for p in self._interpolate(words[len(word_alignments):], last_end, end_s):
                    p["prob"] = 0.3
                    word_alignments.append(p)
            return word_alignments
        return None

    def _try_argmax_alignment(self, logits, words: list[str], start_s: float, end_s: float) -> list[dict] | None:
        import torch

        predicted_ids = torch.argmax(logits, dim=-1)
        tokens = self._processor.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        max_probs = log_probs.max(dim=-1).values
        frame_duration = (end_s - start_s) / len(tokens)

        word_alignments = []
        word_idx = 0
        word_start = None
        word_scores = []

        for idx, token in enumerate(tokens):
            if token in ("<pad>", "|"):
                if word_start is not None and token == "|":
                    word_alignments.append({
                        "word": words[word_idx],
                        "start": word_start,
                        "end": start_s + idx * frame_duration,
                        "prob": float(np.exp(np.mean(word_scores))) if word_scores else 0.5,
                    })
                    word_idx += 1
                    if word_idx >= len(words):
                        break
                    word_scores = []
                word_start = None
                continue
            if word_start is None:
                word_start = start_s + idx * frame_duration
            word_scores.append(max_probs[idx].item())

        if word_start is not None and word_idx < len(words):
            word_alignments.append({
                "word": words[word_idx],
                "start": word_start,
                "end": end_s,
                "prob": float(np.exp(np.mean(word_scores))) if word_scores else 0.5,
            })

        if len(word_alignments) >= len(words) * 0.7:
            if len(word_alignments) < len(words):
                last_end = word_alignments[-1]["end"] if word_alignments else start_s
                for p in self._interpolate(words[len(word_alignments):], last_end, end_s):
                    p["prob"] = 0.3
                    word_alignments.append(p)
            return word_alignments
        return None

    def _interpolate(self, words: list[str], start_s: float, end_s: float) -> list[dict]:
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            return []
        char_duration = (end_s - start_s) / total_chars
        current = start_s
        results = []
        for w in words:
            w_dur = len(w) * char_duration
            results.append({"word": w, "start": current, "end": current + w_dur, "prob": 0.1})
            current += w_dur
        return results

    def cleanup(self) -> None:
        del self._model, self._processor, self._waveform, self._full_logits
        self._model = self._processor = self._waveform = self._full_logits = None
        gc.collect()


def _longest_common_subsequence_words(left: list[dict], right: list[dict]) -> int:
    """
    Find the length of the longest common word subsequence between the tail of
    `left` and the head of `right` in the overlap zone.

    Returns the number of matching words from the *end* of left that match
    the *start* of right. Used for seam merging.
    """
    left_words = [w["word"].lower().strip(".,!?;:") for w in left]
    right_words = [w["word"].lower().strip(".,!?;:") for w in right]

    best_length = 0
    
    max_check = min(len(left_words), len(right_words), 30)  
    for length in range(1, max_check + 1):
        left_tail = left_words[-length:]
        right_head = right_words[:length]
        if left_tail == right_head:
            best_length = length

    return best_length


def _merge_chunk_words_at_seam(
    prev_words: list[dict],
    curr_words: list[dict],
) -> list[dict]:
    """
    Merge two overlapping word lists at the seam.

    Strategy: find the longest common word subsequence between the tail of
    prev_words and the head of curr_words. Keep prev up to the match point,
    curr from after the match point.
    """
    if not prev_words or not curr_words:
        return prev_words + curr_words

    match_len = _longest_common_subsequence_words(prev_words, curr_words)

    if match_len > 0:
        
        merged = prev_words + curr_words[match_len:]
        console.print(f"        [dim]Seam merge: {match_len} words matched at boundary[/dim]")
    else:
        
        merged = prev_words + curr_words
        console.print(f"        [dim]Seam merge: no word match, concatenating[/dim]")

    return merged


def _run_asr_chunked(audio_path: Path, db: ProjectDB, speech_timestamps: list[dict], chunks: list[tuple[float, float]]) -> None:
    """
    ASR transcription fanned out by scene-aligned chunks (map-reduce) with
    overlap windows and seam merging for long videos.

    Each chunk gets +8s overlap on each side. After transcription, overlapping
    regions are merged using longest-common-word-subsequence matching.
    """
    overlap_s = CFG.audio.chunk_overlap_s
    console.print(f"    [dim](B2) ASR + CTC alignment (Map-Reduce, {len(chunks)} chunks, ±{overlap_s}s overlap)...[/dim]")

    from faster_whisper import WhisperModel
    import librosa
    cfg = CFG.audio
    model = WhisperModel(
        cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    total_audio_s = len(y) / sr
    hotwords = "antigravity, trim, timeline, codec, rendering, louder, Umm, uh, you know, like..."

    
    chunk_results: list[dict] = []  

    for chunk_idx, (c_start, c_end) in enumerate(chunks):
        
        pad_start = max(0.0, c_start - overlap_s)
        pad_end = min(total_audio_s, c_end + overlap_s)

        start_sample = int(pad_start * sr)
        end_sample = int(pad_end * sr)
        chunk_y = y[start_sample:end_sample]

        if len(chunk_y) < sr:
            continue

        segments, info = model.transcribe(
            chunk_y,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=hotwords,
        )

        chunk_segs = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            seg_start = pad_start + segment.start
            seg_end = pad_start + segment.end
            avg_logprob = getattr(segment, "avg_logprob", 0.0)

            
            if avg_logprob < cfg.whisper_beam_escalation_threshold:
                console.print(f"      [dim]Chunk {chunk_idx}: low logprob ({avg_logprob:.2f}), re-transcribing with beam={cfg.whisper_beam_escalation_size}...[/dim]")
                try:
                    seg_start_sample = int(segment.start * sr)
                    seg_end_sample = int(segment.end * sr)
                    seg_audio = chunk_y[seg_start_sample:seg_end_sample]
                    if len(seg_audio) > sr * 0.3:
                        retry_segs, _ = model.transcribe(
                            seg_audio,
                            word_timestamps=True,
                            vad_filter=False,
                            beam_size=cfg.whisper_beam_escalation_size,
                            best_of=cfg.whisper_beam_escalation_size,
                        )
                        retry_list = list(retry_segs)
                        if retry_list:
                            best = retry_list[0]
                            if getattr(best, "avg_logprob", -999) > avg_logprob:
                                text = best.text.strip() or text
                                avg_logprob = getattr(best, "avg_logprob", avg_logprob)
                except Exception:
                    pass  

            offset_words = []
            if segment.words:
                for w in segment.words:
                    offset_words.append(w._replace(
                        start=pad_start + w.start,
                        end=pad_start + w.end
                    ))

            chunk_segs.append({
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "words": offset_words,
                "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                "avg_logprob": avg_logprob,
            })

        chunk_results.append({
            "segments": chunk_segs,
            "c_start": c_start,
            "c_end": c_end,
            "pad_start": pad_start,
            "pad_end": pad_end,
        })

    del model
    gc.collect()

    
    
    all_segments = []
    utt_id = 0

    for cr in chunk_results:
        for seg in cr["segments"]:
            
            seg_mid = (seg["start"] + seg["end"]) / 2
            if seg_mid < cr["c_start"] or seg_mid > cr["c_end"]:
                continue  
            seg["id"] = utt_id
            all_segments.append(seg)
            utt_id += 1

    clean_segments = _apply_hallucination_guards(all_segments, speech_timestamps)
    video = db.get_video()
    total_duration = video["duration_s"] if video else 99999.0

    
    aligner = _CTCAligner(audio_path)

    for seg in clean_segments:
        start_c = max(0.0, min(seg["start"], total_duration))
        end_c = max(0.0, min(seg["end"], total_duration))
        db.insert_utterance(seg["id"], start_c, end_c, seg["text"])

        aligned_words, snap_tol = aligner.align_utterance(seg["text"], start_c, end_c)

        for word_idx, w in enumerate(aligned_words):
            w_start = max(0.0, min(w["start"], total_duration))
            w_end = max(0.0, min(w["end"], total_duration))
            db.insert_word(
                utt_id=seg["id"],
                idx=word_idx,
                word=w["word"],
                start=w_start,
                end=w_end,
                prob=w.get("prob"),
                snap_tolerance=snap_tol,
            )

    aligner.cleanup()
    db.set_coverage("transcript", "available")
    db.set_model_manifest("asr", f"faster-whisper-{cfg.whisper_model_size}", cfg.whisper_model_size)


def _run_asr(audio_path: Path, db: ProjectDB, speech_timestamps: list[dict]) -> None:
    """ASR transcription with real beam-search escalation and batched CTC forced alignment."""
    console.print("    [dim](B2) ASR + CTC forced alignment...[/dim]")

    try:
        from faster_whisper import WhisperModel
        import librosa
        cfg = CFG.audio
        model = WhisperModel(
            cfg.whisper_model_size,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )

        hotwords = "antigravity, trim, timeline, codec, rendering, louder, Umm, uh, you know, like..."

        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=hotwords,
        )

        console.print(f"    Language: {info.language} (prob: {info.language_probability:.2f})")

        
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        
        utt_id = 0
        all_segments = []
        escalation_count = 0

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            avg_logprob = getattr(segment, "avg_logprob", 0.0)

            
            if avg_logprob < cfg.whisper_beam_escalation_threshold:
                console.print(f"      [dim]Low logprob ({avg_logprob:.2f}), escalating to beam={cfg.whisper_beam_escalation_size}...[/dim]")
                try:
                    seg_start_sample = int(segment.start * sr)
                    seg_end_sample = int(segment.end * sr)
                    seg_audio = y[seg_start_sample:seg_end_sample]
                    if len(seg_audio) > sr * 0.3:
                        retry_segs, _ = model.transcribe(
                            seg_audio,
                            word_timestamps=True,
                            vad_filter=False,
                            beam_size=cfg.whisper_beam_escalation_size,
                            best_of=cfg.whisper_beam_escalation_size,
                        )
                        retry_list = list(retry_segs)
                        if retry_list:
                            best = retry_list[0]
                            if getattr(best, "avg_logprob", -999) > avg_logprob:
                                text = best.text.strip() or text
                                avg_logprob = getattr(best, "avg_logprob", avg_logprob)
                                escalation_count += 1
                except Exception:
                    pass  

            all_segments.append({
                "id": utt_id,
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "words": segment.words or [],
                "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                "avg_logprob": avg_logprob,
            })
            utt_id += 1

        if escalation_count > 0:
            console.print(f"    Beam escalation improved {escalation_count} segments")

        del model, y
        import gc
        gc.collect()
    except Exception as e:
        console.print(f"    [yellow]⚠ ASR failed ({e}). Active Fallback: CPU + tiny.en model.[/yellow]")
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)
            all_segments = []
            utt_id = 0
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                all_segments.append({
                    "id": utt_id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                    "words": segment.words or [],
                    "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                    "avg_logprob": getattr(segment, "avg_logprob", 0.0),
                })
                utt_id += 1
            del model
            import gc
            gc.collect()
        except Exception as e2:
            import logging
            logging.getLogger("trim").warning(f"ASR completely failed even on CPU: {e2}. Yielding empty transcript.")
            all_segments = []

    clean_segments = _apply_hallucination_guards(all_segments, speech_timestamps)
    video = db.get_video()
    total_duration = video["duration_s"] if video else 99999.0

    
    aligner = _CTCAligner(audio_path)

    for seg in clean_segments:
        start_c = max(0.0, min(seg["start"], total_duration))
        end_c = max(0.0, min(seg["end"], total_duration))
        db.insert_utterance(seg["id"], start_c, end_c, seg["text"])

        
        aligned_words, snap_tol = aligner.align_utterance(seg["text"], start_c, end_c)

        for word_idx, w in enumerate(aligned_words):
            w_start = max(0.0, min(w["start"], total_duration))
            w_end = max(0.0, min(w["end"], total_duration))
            db.insert_word(
                utt_id=seg["id"],
                idx=word_idx,
                word=w["word"],
                start=w_start,
                end=w_end,
                prob=w.get("prob"),
                snap_tolerance=snap_tol,
            )

    aligner.cleanup()
    db.set_coverage("transcript", "available")
    db.set_model_manifest("asr", f"faster-whisper-{cfg.whisper_model_size}", cfg.whisper_model_size)


def _apply_hallucination_guards(
    segments: list[dict],
    speech_timestamps: list[dict],
) -> list[dict]:
    clean = []
    speech_intervals = [(ts["start"], ts["end"]) for ts in speech_timestamps]
    gratitude_patterns = re.compile(
        r"(thanks?\s+for\s+watching|please\s+subscribe|like\s+and\s+subscribe|"
        r"don'?t\s+forget\s+to\s+subscribe|see\s+you\s+next\s+time|"
        r"thank\s+you\s+for\s+listening)",
        re.IGNORECASE,
    )

    for seg in segments:
        text = seg["text"]
        words = text.lower().split()
        if len(words) >= 6:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
            counter = Counter(trigrams)
            max_repeat = max(counter.values()) if counter else 0
            if max_repeat >= 4:
                continue

        if gratitude_patterns.search(text):
            seg_in_speech = any(
                seg["start"] < s_end and seg["end"] > s_start
                for s_start, s_end in speech_intervals
            )
            if not seg_in_speech:
                continue

        no_speech = seg.get("no_speech_prob", 0)
        if no_speech > 0.7:
            seg_in_speech = any(
                seg["start"] < s_end and seg["end"] > s_start
                for s_start, s_end in speech_intervals
            )
            if not seg_in_speech:
                continue

        clean.append(seg)
    return clean






def _run_filler_detection(db: ProjectDB) -> None:
    console.print("    [dim](B4) Filler detection...[/dim]")
    cfg = CFG.audio
    words = db.get_words()
    utterances = db.get_utterances()

    utt_map = {u["id"]: u for u in utterances}
    speaker_word_counts: dict[str, int] = defaultdict(int)
    speaker_filler_counts: dict[str, int] = defaultdict(int)

    always_pattern = re.compile(
        r"^(" + "|".join(re.escape(f) for f in cfg.filler_words_always) + r")$",
        re.IGNORECASE,
    )
    contextual_pattern = re.compile(
        r"^(" + "|".join(re.escape(f) for f in cfg.filler_words_contextual) + r")$",
        re.IGNORECASE,
    )

    filler_count = 0
    for w in words:
        utt = utt_map.get(w["utt_id"])
        speaker = utt.get("speaker_id", "unknown") if utt else "unknown"
        speaker_word_counts[speaker] += 1
        word_text = w["word"].strip().lower()
        if contextual_pattern.match(word_text):
            speaker_filler_counts[speaker] += 1

    speaker_filler_rates = {}
    for speaker, count in speaker_word_counts.items():
        filler_c = speaker_filler_counts.get(speaker, 0)
        speaker_filler_rates[speaker] = filler_c / max(count, 1)

    for w in words:
        word_text = w["word"].strip().lower()
        utt = utt_map.get(w["utt_id"])
        speaker = utt.get("speaker_id", "unknown") if utt else "unknown"

        if always_pattern.match(word_text):
            db.insert_filler(word_text, w["start_time"], w["end_time"], confidence=0.95)
            filler_count += 1
        elif contextual_pattern.match(word_text):
            baseline_rate = speaker_filler_rates.get(speaker, 0)
            confidence = max(0.3, 0.7 - baseline_rate * 2.0)
            db.insert_filler(word_text, w["start_time"], w["end_time"], confidence=confidence)
            filler_count += 1

    # §1.6: N-gram filler matching for multi-word phrases
    _NGRAM_FILLERS = [
        "you know", "i mean", "sort of", "kind of", "you know what i mean",
        "like i said", "to be honest", "at the end of the day",
    ]
    for n in (3, 2):  # check trigrams first, then bigrams
        for idx in range(len(words) - n + 1):
            span = words[idx:idx + n]
            phrase = " ".join(w["word"].strip().lower() for w in span)
            if phrase in _NGRAM_FILLERS:
                # Don't double-count if individual words already tagged
                span_start = span[0]["start_time"]
                span_end = span[-1]["end_time"]
                db.insert_filler(phrase, span_start, span_end, confidence=0.80)
                filler_count += 1

    console.print(f"    Fillers detected: {filler_count}")
    db.set_coverage("fillers", "available")






def _track_beats_and_downbeats(y: np.ndarray, sr: int, music_regions: list[tuple[float, float]], db: ProjectDB) -> None:
    """Run beat tracking on music regions with tempo confidence rejection."""
    import librosa
    beat_count = 0
    for region_start, region_end in music_regions:
        start_sample = int(region_start * sr)
        end_sample = int(region_end * sr)
        region_audio = y[start_sample:end_sample]

        if len(region_audio) < sr:
            continue

        tempo, beat_frames = librosa.beat.beat_track(y=region_audio, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        
        
        try:
            onset_env = librosa.onset.onset_strength(y=region_audio, sr=sr)
            ac = librosa.autocorrelate(onset_env, max_size=onset_env.shape[0])
            ac_norm = ac / (ac[0] + 1e-10)
            
            bpm_range_samples = [int(60.0 / bpm * sr / 512) for bpm in [60, 200] if bpm > 0]
            if len(bpm_range_samples) >= 2:
                search_range = ac_norm[min(bpm_range_samples):max(bpm_range_samples)]
                tempo_confidence = float(np.max(search_range)) if len(search_range) > 0 else 0.0
            else:
                tempo_confidence = 0.5
        except Exception:
            tempo_confidence = 0.5

        if tempo_confidence < 0.3:
            console.print(f"      [dim]Beat grid rejected for region [{region_start:.1f}-{region_end:.1f}] (confidence: {tempo_confidence:.2f})[/dim]")
            continue

        for i, bt in enumerate(beat_times):
            is_downbeat = 0  # §4.2: Stop pretending phase-accuracy exists
            db.insert_beat(region_start + float(bt), is_downbeat=is_downbeat)
            beat_count += 1

    console.print(f"    Beats & Downbeats detected: {beat_count}")



_EVENT_LABEL_MAP: dict[str, list[str]] = {
    "music": ["music", "singing", "guitar", "piano", "drum", "musical"],
    "laughter": ["laughter", "giggling", "snicker", "chuckle"],
    "applause": ["applause", "cheering", "clapping"],
    "crowd_noise": ["crowd", "hubbub", "babble", "audience"],
    "door_slam": ["door", "slam", "knock"],
    "keyboard": ["keyboard", "typing", "computer keyboard", "typewriter"],
    "phone_ring": ["telephone", "ringtone", "ring", "bell"],
    "cough": ["cough", "sneeze", "throat"],
    "gasp": ["gasp", "breathing", "sigh", "exhale"],
}


def _classify_event_label(raw_label: str) -> str | None:
    """Map an AST raw label to one of our 9 canonical event labels."""
    raw = raw_label.lower()
    for canonical, keywords in _EVENT_LABEL_MAP.items():
        if any(kw in raw for kw in keywords):
            return canonical
    return None


def _apply_median_filter(sequence: list[float], kernel_size: int = 3) -> list[float]:
    """Apply a median filter to a 1-D confidence sequence."""
    if len(sequence) < kernel_size:
        return sequence
    result = []
    half = kernel_size // 2
    for i in range(len(sequence)):
        start = max(0, i - half)
        end = min(len(sequence), i + half + 1)
        result.append(float(np.median(sequence[start:end])))
    return result


def _apply_hysteresis(
    times: list[float],
    confidences: list[float],
    enter_threshold: float = 0.5,
    exit_threshold: float = 0.3,
    hop_s: float = 1.0,
) -> list[tuple[float, float, float]]:
    """
    Hysteresis thresholding: enter event at confidence > enter_threshold,
    exit at < exit_threshold. Prevents flickering.

    Returns list of (start_time, end_time, mean_confidence) event regions.
    """
    events = []
    in_event = False
    event_start = 0.0
    event_confs = []

    for t, conf in zip(times, confidences):
        if not in_event and conf > enter_threshold:
            in_event = True
            event_start = t
            event_confs = [conf]
        elif in_event:
            if conf < exit_threshold:
                in_event = False
                event_end = t + hop_s
                mean_conf = float(np.mean(event_confs))
                events.append((event_start, event_end, mean_conf))
                event_confs = []
            else:
                event_confs.append(conf)

    
    if in_event and event_confs:
        events.append((event_start, times[-1] + hop_s, float(np.mean(event_confs))))

    return events


def _run_audio_events(audio_path: Path, db: ProjectDB) -> None:
    """
    Classify audio events using AST with:
    - 9 canonical event labels
    - 2s sliding window with 1s hop (50% overlap)
    - 3-frame median filter on per-class confidence
    - Hysteresis thresholding (enter > 0.5, exit < 0.3)
    """
    console.print("    [dim](B5) Audio events (9 labels, median+hysteresis)...[/dim]")

    event_count = 0
    music_regions = []
    try:
        from transformers import pipeline
        import librosa
        
        classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
        
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        window_s = 2.0
        hop_s = 1.0  
        chunk_len = int(window_s * sr)
        hop_len = int(hop_s * sr)

        
        class_confs: dict[str, list[float]] = {label: [] for label in _EVENT_LABEL_MAP}
        time_stamps: list[float] = []

        for start_sample in range(0, len(y) - chunk_len, hop_len):
            chunk = y[start_sample:start_sample + chunk_len]
            t_center = (start_sample + chunk_len / 2) / sr
            time_stamps.append(t_center)

            results = classifier(chunk, top_k=5)

            
            frame_scores: dict[str, float] = {label: 0.0 for label in _EVENT_LABEL_MAP}
            for res in results:
                canonical = _classify_event_label(res["label"])
                if canonical:
                    frame_scores[canonical] = max(frame_scores[canonical], res["score"])

            for label in _EVENT_LABEL_MAP:
                class_confs[label].append(frame_scores[label])

        
        for label, confs in class_confs.items():
            if not confs:
                continue

            
            filtered = _apply_median_filter(confs, kernel_size=3)

            
            events = _apply_hysteresis(
                time_stamps, filtered,
                enter_threshold=0.5,
                exit_threshold=0.3,
                hop_s=hop_s,
            )

            for ev_start, ev_end, ev_conf in events:
                db.insert_audio_event(label, ev_start, ev_end, confidence=ev_conf)
                event_count += 1
                if label == "music":
                    music_regions.append((ev_start, ev_end))

        console.print(f"    Audio events detected: {event_count}")

        if music_regions:
            _track_beats_and_downbeats(y, sr, music_regions, db)

        db.set_coverage("audio_events", "available")
    except Exception:
        
        _run_audio_events_heuristic(audio_path, db)


def _run_audio_events_heuristic(audio_path: Path, db: ProjectDB) -> None:
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        hop_length = 512
        frame_length = 2048
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=frame_length)[0]
        times = librosa.frames_to_time(np.arange(len(flatness)), sr=sr, hop_length=hop_length)

        threshold = CFG.audio.spectral_flatness_threshold
        in_music = False
        music_start = 0.0
        music_regions = []

        for idx, (t, f) in enumerate(zip(times, flatness)):
            if f < threshold and not in_music:
                in_music = True
                music_start = t
            elif f >= threshold and in_music:
                in_music = False
                if t - music_start > 2.0:
                    db.insert_audio_event("music", music_start, t, confidence=0.6)
                    music_regions.append((music_start, t))

        if music_regions:
            _track_beats_and_downbeats(y, sr, music_regions, db)

        # §1.9: Honest coverage — heuristic can only detect music, not laughter/applause
        db.set_coverage("audio_events", "music_only_heuristic")
    except ImportError:
        db.set_coverage("audio_events", "unavailable")






def _run_speaker_attribution(audio_path: Path, db: ProjectDB) -> None:
    """Cluster utterances and resolve identities against the workspace speaker registry."""
    console.print("    [dim](B6) Speaker attribution...[/dim]")

    try:
        import librosa
        import torch
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from speechbrain.inference.speaker import EncoderClassifier

        utterances = db.get_utterances()
        if len(utterances) < 2:
            return

        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        features = []
        valid_utts = []

        # §4.1: Use SpeechBrain ECAPA-TDNN for robust speaker embeddings
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

        for utt in utterances:
            start_sample = int(utt["start_time"] * sr)
            end_sample = int(utt["end_time"] * sr)
            segment = y[start_sample:end_sample]

            if len(segment) < sr * 0.3:
                continue

            # Compute ECAPA embedding
            signal = torch.tensor(segment).unsqueeze(0)
            feat = classifier.encode_batch(signal).squeeze().detach().numpy()
            
            features.append(feat)
            valid_utts.append(utt["id"])

        if len(features) < 2:
            return

        X = np.array(features)
        best_labels = None
        best_score = -1
        best_n = 2

        for n in range(CFG.audio.min_speakers, min(CFG.audio.max_speakers + 1, len(X))):
            try:
                clustering = AgglomerativeClustering(n_clusters=n)
                labels = clustering.fit_predict(X)
                score = silhouette_score(X, labels) if n < len(X) else -1
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_n = n
            except Exception:
                continue

        if best_labels is not None:
            cluster_embeddings = defaultdict(list)
            for utt_id, label, feat in zip(valid_utts, best_labels, features):
                cluster_embeddings[label].append(feat)

            prior_speakers = db.get_speaker_embeddings()
            resolved_names = {}

            for label, feats in cluster_embeddings.items():
                centroid = np.mean(feats, axis=0).astype(np.float32)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

                best_match = None
                best_sim = -1.0

                for prior in prior_speakers:
                    prior_emb = np.frombuffer(prior["embedding"], dtype=np.float32)
                    prior_norm = prior_emb / (np.linalg.norm(prior_emb) + 1e-10)
                    sim = float(np.dot(centroid_norm, prior_norm))
                    if sim > best_sim:
                        best_sim = sim
                        best_match = prior["speaker_id"]

                if best_sim > 0.85 and best_match:
                    resolved_names[label] = best_match
                else:
                    resolved_names[label] = f"speaker_{label}"
                    
                    db.insert_speaker_embedding(
                        speaker_id=f"speaker_{label}",
                        embedding=centroid.tobytes(),
                        dim=len(centroid)
                    )

            for utt_id, label in zip(valid_utts, best_labels):
                db.update_utterance(utt_id, speaker_id=resolved_names[label])

            db.set_coverage("speakers", "available")
        else:
            db.set_coverage("speakers", "unavailable")

        del y
        gc.collect()
    except Exception:
        db.set_coverage("speakers", "unavailable")






def _run_loudness_curve(audio_path: Path, db: ProjectDB) -> None:
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        window_samples = int(CFG.audio.loudness_window_ms / 1000.0 * sr)
        hop_samples = window_samples

        for start in range(0, len(y) - window_samples, hop_samples):
            window = y[start:start + window_samples]
            rms = np.sqrt(np.mean(window ** 2))
            rms_db = 20 * np.log10(max(rms, 1e-10))
            t_center = (start + window_samples / 2) / sr
            db.insert_loudness_sample(t_center, round(rms_db, 2))

        db.set_coverage("loudness", "available")
    except Exception:
        db.set_coverage("loudness", "unavailable")






def _run_topic_segmentation(db: ProjectDB) -> None:
    from trim_engine.llm import call_structured
    from trim_engine.schemas import TopicSegmentationResponse

    utterances = db.get_utterances()
    if not utterances:
        return

    transcript_lines = []
    for utt in utterances:
        speaker = utt.get("speaker_id") or "unknown"
        transcript_lines.append(f"[UTT {utt['id']}] ({speaker}): {utt['text']}")

    try:
        response = call_structured(
            prompt_name="topic_segmenter",
            user_content="\n".join(transcript_lines),
            schema=TopicSegmentationResponse,
            effort="low",
            db=db
        )
    except Exception as e:
        from trim_engine.cli import console
        console.print(f"      [yellow]⚠ Topic segmentation degraded: {e}[/yellow]")
        db.set_coverage("topics", "unavailable", note=str(e))
        return

    for seg in response.segments:
        if not seg.utterance_ids:
            continue
        seg_utts = [u for u in utterances if u["id"] in seg.utterance_ids]
        if not seg_utts:
            continue

        start = min(u["start_time"] for u in seg_utts)
        end = max(u["end_time"] for u in seg_utts)
        topic_id = db.insert_topic(seg.topic_label, seg.topic_class, start, end)

        for utt_id in seg.utterance_ids:
            # §3.2: Assign dialogue act per-utterance
            d_act = seg.dialogue_acts.get(utt_id) or "statement"
            db.update_utterance(utt_id, topic_id=topic_id, dialogue_act=d_act)
    db.set_coverage("topics", "available")






def _run_retake_detection(db: ProjectDB) -> None:
    cfg = CFG.audio
    utterances = db.get_utterances()
    if len(utterances) < 2:
        return

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CFG.embedding.text_model_name)
        embeddings = model.encode([u["text"] for u in utterances], normalize_embeddings=True)
    except Exception:
        return

    cluster_id = 0
    used = set()

    for i in range(len(utterances)):
        if i in used:
            continue

        cluster_members = [i]
        for j in range(i + 1, len(utterances)):
            if j in used:
                continue

            sim = float(np.dot(embeddings[i], embeddings[j]))
            time_gap = abs(utterances[j]["start_time"] - utterances[i]["start_time"])
            # §1.5: Proper window — skip if too close (continuation) OR too far (unrelated)
            if sim <= cfg.retake_similarity_threshold or time_gap < cfg.retake_min_gap_s or time_gap > cfg.retake_max_gap_s:
                continue

            speaker_i = utterances[i].get("speaker_id", "")
            speaker_j = utterances[j].get("speaker_id", "")
            if speaker_i and speaker_j and speaker_i != speaker_j:
                continue

            dur_i = utterances[i]["end_time"] - utterances[i]["start_time"]
            dur_j = utterances[j]["end_time"] - utterances[j]["start_time"]
            if dur_i > 0 and dur_j > 0:
                if min(dur_i, dur_j) / max(dur_i, dur_j) < cfg.retake_duration_ratio_threshold:
                    continue

            cluster_members.append(j)
            used.add(j)

        if len(cluster_members) > 1:
            used.add(i)
            members_sorted = sorted(cluster_members, key=lambda x: utterances[x]["start_time"])
            for take_idx, member_idx in enumerate(members_sorted):
                db.insert_retake(cluster_id, utterances[member_idx]["id"], take_idx)
            cluster_id += 1
    db.set_coverage("retakes", "available" if cluster_id > 0 else "unavailable")






def _run_breath_detection(audio_path: Path, db: ProjectDB) -> None:
    """Detect breaths using Respiro-en, with ZCR+energy heuristic fallback."""
    console.print("    [dim](A1) Breath detection...[/dim]")
    
    # Attempt to load Respiro-en
    respiro_model_path = Path(__file__).parent.parent / "models" / "respiro_en" / "respiro-en.pt"
    if respiro_model_path.exists():
        try:
            import sys
            respiro_dir = str(respiro_model_path.parent)
            if respiro_dir not in sys.path:
                sys.path.insert(0, respiro_dir)
            
            from modules import DetectionNet, BreathDetector
            import torch
            
            console.print("      [dim]Using Respiro-en deep model...[/dim]")
            device = torch.device("cpu")
            model = DetectionNet().to(device)
            model.load_state_dict(torch.load(str(respiro_model_path), map_location=device))
            model.eval()
            
            detector = BreathDetector(model, device=device)
            # The detector returns an IntervalTree of breaths
            tree = detector(str(audio_path))
            
            # Insert into database
            for interval in tree:
                db.insert_breath(interval.begin, interval.end)
                
            db.set_coverage("breaths", "respiro")
            return
        except Exception as e:
            console.print(f"      [yellow]⚠ Respiro-en failed ({e}), falling back to heuristic...[/yellow]")
    else:
        console.print("      [yellow]⚠ Respiro-en model not found, falling back to heuristic...[/yellow]")

    console.print("      [dim]Using ZCR+Energy heuristic...[/dim]")
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000)
    
    # Simple heuristic: breaths are low energy, but have high ZCR compared to pure silence.
    # We look at non-speech regions adjacent to speech.
    
    frame_length = 2048
    hop_length = 512
    
    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    
    times = librosa.frames_to_time(np.arange(len(rmse)), sr=sr, hop_length=hop_length)
    
    silences = db.get_silences()
    breath_count = 0
    
    # We only look for breaths inside silences (non-speech)
    for silence in silences:
        start_t = silence["start_time"]
        end_t = silence["end_time"]
        
        # Filter frames within this silence
        mask = (times >= start_t) & (times <= end_t)
        if not np.any(mask):
            continue
            
        region_times = times[mask]
        region_rmse = rmse[mask]
        region_zcr = zcr[mask]
        
        if len(region_rmse) < 3:
            continue
            
        # Breath characteristics: Energy bump but still low, moderate ZCR
        # Normalize RMSE in this silence
        norm_rmse = region_rmse / (np.max(region_rmse) + 1e-6)
        
        # Find peaks in energy that might be breaths
        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(norm_rmse, height=0.2, distance=5)
        
        for p in peaks:
            # Check if ZCR is indicative of breath (white noise-like, higher than hum, lower than fricatives)
            if 0.05 < region_zcr[p] < 0.3:
                breath_start = max(start_t, region_times[p] - 0.2)
                breath_end = min(end_t, region_times[p] + 0.3)
                db.insert_breath(breath_start, breath_end, confidence=0.7)
                breath_count += 1
                
    console.print(f"    Detected {breath_count} breaths.")
    db.set_coverage("breaths", "heuristic")

def run_audio_intelligence(project_dir: Path, db: ProjectDB) -> None:
    audio_path = project_dir / "audio.wav"
    if not audio_path.exists():
        raise RuntimeError("audio.wav not found")

    video = db.get_video()
    total_duration = video["duration_s"] if video else 0.0

    
    from trim_engine.ingest.orchestrator import _get_scene_aligned_chunks
    chunks = _get_scene_aligned_chunks(total_duration, db)

    speech_timestamps = _run_vad(audio_path, db)
    
    if len(chunks) > 1:
        _run_asr_chunked(audio_path, db, speech_timestamps, chunks)
    else:
        _run_asr(audio_path, db, speech_timestamps)

    # §1.6: Speaker attribution runs BEFORE filler detection so fillers have speaker context
    _run_speaker_attribution(audio_path, db)
    _run_filler_detection(db)
    _run_breath_detection(audio_path, db)
    _run_audio_events(audio_path, db)
    _run_loudness_curve(audio_path, db)
    _run_topic_segmentation(db)
    _run_retake_detection(db)
