# Edit Report — v22

**Prompt:** "Keep only questions."

## Summary

| Metric | Value |
|--------|-------|
| Duration (before) | 80.8s (1.3 min) |
| Duration (after) | 80.7s (1.3 min) |
| Reduction | 0.1% |
| Removals | 0 |
| Critic | FAIL: 1 issues |
| LLM Cost | $2.0552 |

## ⚠ Unsatisfied Operations

- Op 0: Op 0 is a hard keep_only directive for 'segments containing spoken questions.' The entire video (0.0s–80.8s) was retained at a 0.1% removal ratio. Reviewing the kept transcript, the only identifiable spoken question is 'Why are balloons so expensive?' (the balloon joke setup near the end). All other content — the opening provocation, the Facebook/internet commentary, the attention span declaration, the TED talk length bit — contains no spoken questions and should have been removed. The keep_only rule was effectively not applied.

## 🔗 Continuity Warnings

- Potential coherence issues detected in kept content
- The keep_only operation was ignored entirely, resulting in the full video being preserved. Only the segment containing the spoken question 'Why are balloons so expensive?' should remain. Op 1 (filler removal) cannot be properly evaluated because Op 0 was never executed — re-run from the story stage so the keep_only selection is applied first before filler removal is assessed on the surviving clips.

## 👤 Profile Preferences Applied

- remove_fillers
- pacing
