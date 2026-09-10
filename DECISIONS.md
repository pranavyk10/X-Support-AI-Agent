# Decision log

Non-obvious choices and why.

1. **Brand = TwitterSupport (X Support)** — User choice; dataset still uses the pre-rebrand handle. Framed explicitly so graders aren’t confused by “X” vs “Twitter”.

2. **Accept thin real volume (~1,257 pairs)** — Unlike Amazon/Apple, TwitterSupport is small. Rather than switching brands, we documented thinness and augmented retrieval with labelled synthetic examples.

3. **Synthetic augmentation only for retrieval, not golden** — Golden set stays on real Kaggle tweets so metrics aren’t scored on our own templates.

4. **11 intents, not 77** — Banking77-style granularity doesn’t match vague TwitterSupport traffic; small set is easier to audit and escalate.

5. **`thanks_or_other` includes vague “help me”** — Modal inbound pattern; forcing these into account/app buckets created false confidence.

6. **Priority-keyword classify (not pure hit-count)** — Hit-count mislabelled abuse tweets containing “campaign” / “timeline”. Priority (safety before ads/app) matches how a human triage agent would break ties.

7. **Escalate-positive as the metric class** — False auto-handle on suspension/abuse is worse than extra human load; optimize recall, report precision honestly.

8. **RAG over fine-tuning** — With ~1k real pairs and a free-tier LLM, retrieval+prompting is more trustworthy and easier to cite evidence for.

9. **Offline agent draft = structured rewrite, not raw NN** — Differentiates agent from the simple baseline without requiring Groq for the 15-minute reproduce path.

10. **Groq model split: 8B classify / 70B draft+judge** — Respects free-tier RPD (70B ≈ 1k/day); 8B handles bulk cheaply.

11. **Cache-first evaluation (`--use-cache`)** — Graders must reproduce headline numbers without burning quota or waiting on rate limits.

12. **Exclude golden `pair_id` from retrieval** — Prevents trivial self-match leakage (still imperfect when templates duplicate).

13. **Heuristic judge fallback when no API key** — Keeps the harness runnable; REPORT calls out that live Groq judging is the stronger signal.

14. **Skip Banking77** — Transfer would cost time for limited TwitterSupport vocabulary overlap; better spent on golden set + failure analysis.

15. **Never invent account actions in prompts** — Brand historically DMs / asks for detail; over-promising is the main safety failure mode for a support LLM.
