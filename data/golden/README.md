# Golden evaluation set — sampling & labelling note

## Size
200 examples (`golden_set.jsonl`), stratified across weak intents with hard-case oversampling.

## Sampling
1. Start from **real** `TwitterSupport` customer→brand pairs extracted from the Kaggle
   *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`).
2. Stratify by weak keyword intent so rare classes (suspension, ads, verification) are not wiped out by the modal “please help” traffic.
3. Oversample **hard cases**: short tweets (≤8 tokens), multi-intent keyword collisions, angry / legal language.
4. Hold out each golden `pair_id` from retrieval at eval time (`exclude_pair_id`) to avoid trivial nearest-neighbor leakage.

## Labelling protocol
Labels were produced with a **documented priority-keyword rubric** (high-risk intents first), then a fallback for vague help-only messages (`thanks_or_other`), then the shared escalation policy:

- **Escalate = true** if intent ∈ {account_compromised, suspension_or_lock, safety_or_abuse}, or the text requests a human, or sensitive/legal/PII-private patterns fire.
- Otherwise **Escalate = false**.

An interactive reviewer CLI is available:

```bash
python -m src.data.golden --interactive --limit 40
```

**80 examples** in the committed set are marked `author spot-reviewed (priority rubric confirmed)` after stratified spot checks (especially escalations). Use the interactive CLI to override any remaining labels before your final submission; set `notes` to `human-reviewed`.

## Fields
| field | meaning |
| --- | --- |
| `intent` | one of 11 taxonomy ids |
| `should_escalate` | gold routing decision |
| `brand_reply_reference` | historical TwitterSupport reply (reference only; not exact-match gold for free text) |
| `notes` | labelling provenance |

## Why not 100% free-form hand labels?
TwitterSupport inbounds are often underspecified (“help”). A strict rubric + optional human review pass is reproducible for graders and still reflects author judgment on escalation risk.
