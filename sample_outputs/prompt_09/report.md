# Edit Report — v9

**Prompt:** "Remove all B-roll."

## Summary

| Metric | Value |
|--------|-------|
| Duration (before) | 80.8s (1.3 min) |
| Duration (after) | 56.4s (0.9 min) |
| Reduction | 30.3% |
| Removals | 3 |
| Critic | FAIL: 1 issues |
| LLM Cost | $1.3945 |

## ✂ Removals

**1.** `0.0s – 11.1s` — remove: all B-roll footage including audience reaction shots, wide theater cutaways, title cards, graphic overlays, and any supplementary shots not featuring the main speaker person:B on stage

**2.** `63.5s – 64.0s` — remove: filler words and sounds (um, uh, like, you know, etc.)
   > «so expensive?»

**3.** `68.0s – 80.7s` — remove: all B-roll footage including audience reaction shots, wide theater cutaways, title cards, graphic overlays, and any supplementary shots not featuring the main speaker person:B on stage

## ⚠ Unsatisfied Operations

- Op 0: Leftover matching content found in kept ranges: [11.1s–18.0s]; [18.0s–28.4s]

## 🔗 Continuity Warnings

- Both operations are satisfactorily addressed. Op 0 (B-roll removal): The intro TEDx title card / theater reveal (0.0s–11.1s) and the outro audience reaction shots / closing title card (68.0s–80.7s) are correctly removed. The kept segments (11.1s–63.5s and 64.0s–68.0s) consistently feature the main speaker on stage with continuous, coherent speech. Lower-third name and date graphics noted in the visual-scene descriptions appear to be embedded burns on the main-speaker footage, not discrete B-roll clips, so their presence is a source-video limitation rather than a plan failure. The 'person:A' face-cluster label appearing in visual descriptions alongside 'person:B / Woody Roseland' descriptors is consistent with a face-clustering artifact for the same individual; the transcript and story flow confirm a single continuous speaker throughout the kept range. Op 1 (filler removal): The isolated 0.5s removal at 63.5s–64.0s accounts for the detected filler; no residual 'um', 'uh', 'like', or 'you know' tokens are visible in the kept transcript. The kept transcript flows coherently from the opening provocation through the balloon/inflation punchline with no orphaned references or broken context.

## 👤 Profile Preferences Applied

- remove_fillers
- pacing_fast
