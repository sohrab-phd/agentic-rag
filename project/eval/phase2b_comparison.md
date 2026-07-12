# Phase 2B — Single-Answer Aggregation Bypass Comparison

**Before:** `results_phase1a_rewrite_hardening.json`
**After:** `results_phase2b.json`

## Before vs After

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Pass rate | 56.7% | 56.7% | +0.0% |
| Passed count | 17/30 | 17/30 | +0 |
| Avg latency | 85.51s | 66.43s | -19.08s |
| Aggregation invocations | 30 | 1 | -29 |
| Aggregation-induced failures | 1 | 0 | -1 |

**Aggregation bypassed:** 29 / 30 queries

## Verification

- **No new failures introduced** (no pass→fail regressions vs Phase 1A)
- **Multi-intent behavior:**
  - `multi_001`: 1 sub-answers, aggregation_ran=False, pass True → True
  - `multi_002`: 2 sub-answers, aggregation_ran=True, pass False → False

### `golestan_direct_010`

- Before pass: **True**
- After pass: **True**
- Aggregation ran after: **False**
