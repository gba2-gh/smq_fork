# Phase 8b: gravity- and heading-aware tokens, extension (deferred)

Revision 1 — 4 October 2026. Proposal; not implemented. Runs only after Phase 8a has been read. Parent:
`PHASE8A_GRAVITY_HEADING_TOKENS.md` (motivation, tokenizer definitions, block weighting, gates and reading rules are
defined there and reused unchanged).

## Scope (the parts of the original Phase 8 plan not run in 8a)

| Stage | Work | Units | Estimate |
|---|---|---|---|
| B1 | BABEL-1: the 8a tokenizer(s) worth carrying forward × seeds 111, 222, `context`, span 2; third seed under the same +1.0 rule | 2-3 per tokenizer | ~23 min per unit |
| D | `centre` arm for the carried-forward tokenizer on BABEL-1/2/3 (does contextualization still add value? as Phase 7's P4) | 2-3 per dataset | BABEL-1 ~19 min per unit; BABEL-2/3 ~2 min |
| E | LARa T5g and T5h × 2 seeds (the 2×2 factorial on LARa) | 4 | ~19 min per unit |

- **BABEL-1 is where the heading channel is tested.** Its `turn` class (11% of segments) is the strongest motivation
  for T5h, and 8a cannot test it. Recall of `turn` against T2 at the same seeds is the class-level read.
- Which tokenizers are carried forward is decided from 8a's report, before 8b is implemented, and recorded here.
  That choice uses labels on BABEL-2/3 and LARa, so BABEL-1 is the prospective check (with Phase 7's caveats: BABEL-1
  is not untouched, and 10 recordings are shared with BABEL-2/3).
- BABEL-1 decodes dominate the cost (~20 min each; all reached the OT cap in Phase 7).

## Out of scope (unchanged from the original plan)

- Global translation (absent from the root-centred features).
- Body-size normalization (one question per phase).
- HuGaDB (no joint positions; would need an IMU analogue).
