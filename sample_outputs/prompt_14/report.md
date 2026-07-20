# Edit Report — v14

**Prompt:** "Remove every shot where Person B appears."

## Summary

| Metric | Value |
|--------|-------|
| Duration (before) | 80.8s (1.3 min) |
| Duration (after) | 23.9s (0.4 min) |
| Reduction | 70.5% |
| Removals | 1 |
| Critic | PASS |
| LLM Cost | $1.5567 |

## ✂ Removals

**1.** `11.1s – 68.0s` — remove: every shot where person:B appears; remove: filler words and verbal fillers; remove: filler words and verbal fillers
   > «Wow What an audience, but if I'm being honest, I don't care what you think of my talk. I Don't...»

## 🔗 Continuity Warnings

- Both hard removal operations are satisfied. Op 0: The entire segment containing person:B (the TED speaker, 11.1s–68.0s) has been removed. All kept segments are either branded title cards, wide audience reaction shots, or production credits — none of which contain person:B. Op 1: Filler words are contained within the removed speech segment (11.1s–68.0s); no spoken filler content is present in the kept transcript. Coherence is acceptable: the kept sequence flows logically as a branded intro → audience applause → outro, which is a valid shell structure for a talk where the speaker content has been intentionally excised. No orphaned references or broken context detected in the kept visual-only segments.

## 👤 Profile Preferences Applied

- remove_fillers
- pacing
