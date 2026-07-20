import pytest
from pathlib import Path
from trim_engine.schemas import Timeline, VideoClip, OutputSpec
from trim_engine.query.renderer.execute import _generate_captions

class MockDB:
    def __init__(self, words):
        self.words = words
    def get_words(self):
        return self.words

def test_generate_captions_reordered_timeline(tmp_path):
    # Phase 2.5: verify captions logic supports non-chronological timelines
    words = [
        {"word": "Hello", "start_time": 0.0, "end_time": 1.0},
        {"word": "world", "start_time": 1.0, "end_time": 2.0},
        {"word": "this", "start_time": 5.0, "end_time": 6.0},
        {"word": "is", "start_time": 6.0, "end_time": 7.0},
    ]
    db = MockDB(words)
    
    # We construct a timeline that reorders [5, 7] before [0, 2]
    clips = [
        VideoClip(src_in=5.0, src_out=7.0),
        VideoClip(src_in=0.0, src_out=2.0)
    ]
    timeline = Timeline(
        version=1,
        fps=30.0,
        source="dummy.mp4",
        output=OutputSpec(),
        video_clips=clips,
        audio_clips=[],
        provenance=[],
    )
    
    _generate_captions(timeline, db, tmp_path)
    
    vtt_content = (tmp_path / "output.vtt").read_text()
    
    # The timeline output duration is 4.0s total.
    # The first clip is [5.0 - 7.0]. It should map to [0.0 - 2.0] in output.
    # Therefore, "this" (5-6) -> 0.0-1.0, "is" (6-7) -> 1.0-2.0.
    # The second clip is [0.0 - 2.0]. It should map to [2.0 - 4.0] in output.
    # Therefore, "Hello" (0-1) -> 2.0-3.0, "world" (1-2) -> 3.0-4.0.
    
    assert "00:00:00.000 --> 00:00:04.000" in vtt_content
    assert "this is Hello world" in vtt_content

