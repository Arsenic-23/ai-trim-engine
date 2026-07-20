# Edit Report — v20

**Prompt:** "Keep only close-up reactions."

## Summary

| Metric | Value |
|--------|-------|
| Duration (before) | 80.8s (1.3 min) |
| Duration (after) | 80.7s (1.3 min) |
| Reduction | 0.1% |
| Removals | 0 |
| Critic | FAIL: 1 issues |
| LLM Cost | $1.9461 |

## ⚠ Unsatisfied Operations

- Op 0: Op 0 is a hard 'keep_only' instruction requiring only close-up reaction shots of audience or persons showing emotional responses. Instead, the entire video (0.0s–80.8s) was retained, including: branded title cards, the speaker's full monologue, wide audience shots (not close-ups), and production outro graphics. None of the retained content qualifies as a close-up reaction shot. The only audience reaction content visible in the scene descriptions is a pair of wide auditorium shots (rows of seated attendees applauding), which are explicitly wide shots — not close-up reaction shots. All non-reaction content should have been removed.

## 🔗 Continuity Warnings

- Potential coherence issues detected in kept content
- The keep_only operation was entirely unapplied — the full source video was passed through unchanged at a 0.1% removal ratio. The story agent must filter the timeline to retain only segments containing close-up reaction shots of audience members or other persons displaying visible emotional responses. If no such close-up reaction shots exist in the source material, that should be surfaced as a null-result rather than defaulting to keeping everything.

## 👤 Profile Preferences Applied

- always_remove_fillers
- pacing_fast
