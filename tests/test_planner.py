import pytest
from trim_engine.query.planner import _merge_overlapping_removals

def test_merge_overlapping_removals():
    removals = [
        (1.0, 3.0, "reason A", "ev A"),
        (2.0, 4.0, "reason B", "ev B"),
        (5.0, 6.0, "reason C", "ev C")
    ]
    merged = _merge_overlapping_removals(removals)
    
    # Should merge the first two
    assert len(merged) == 2
    assert merged[0][0] == 1.0
    assert merged[0][1] == 4.0
    assert "reason A" in merged[0][2]
    assert merged[1][0] == 5.0
    assert merged[1][1] == 6.0

def test_merge_overlapping_removals_contained():
    removals = [
        (1.0, 5.0, "reason A", "ev A"),
        (2.0, 3.0, "reason B", "ev B")
    ]
    merged = _merge_overlapping_removals(removals)
    
    assert len(merged) == 1
    assert merged[0][0] == 1.0
    assert merged[0][1] == 5.0
