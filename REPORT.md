# Report: X Support AI Agent

**Brand:** TwitterSupport (X Support, pre-rebrand data)  
**Dataset:** Kaggle *Customer Support on Twitter* (thoughtvector)  
**Eval:** 200-example golden set · trivial + simple baselines · LLM-as-judge rubric

## 1. Problem framing — what “good” means

TwitterSupport inbounds are noisy and often underspecified (“pls help”). A trustworthy agent for this brand should:

1. **Route** into a small, inspectable intent set (11 classes) rather than free-form categories.
2. **Draft** in historical brand voice: short, ask for a DM / more detail, point to help.twitter.com — **never invent** account actions (unsuspend, verify, refund).
3. **Escalate** when risk is high (compromise, suspension, safety/abuse), when the user asks for a human, or when retrieval confidence is weak — with an explicit reason code.

### What we chose not to build
- Multi-turn dialogue state / ticket systems
- Live X/Twitter API integration
- Fine-tuned LLMs on full 3M tweets
- Banking77 intent transfer (skipped; brand language differs enough that keyword+RAG was faster to validate)
- Auto-closing legal / child-safety issues without a human

## 2. System

```
inbound → intent classify → retrieve top-3 historical replies → draft → escalate decision
```

| Stage | Offline (reproducible) | Live (Groq free tier) |
| --- | --- | --- |
| Classify | Priority keyword rules | `llama-3.1-8b-instant` JSON |
| Draft | Structured rewrite of retrieved evidence | `llama-3.3-70b-versatile` grounded JSON |
| Escalate | Shared rule policy (risk intents, human-request, sim/confidence thresholds) | same |
| Retrieve | `all-MiniLM-L6-v2` cosine over real+synthetic index | same |

**Baselines**
- **Trivial:** always `thanks_or_other` + fixed template + never escalate
- **Simple:** keyword classify + raw nearest-neighbor brand reply + escalation rules

## 3. Results vs baselines (golden n=200)

| system | intent acc | intent macro-F1 | escalate F1 | escalate recall | reply emb-sim (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| trivial | 0.39 | 0.06 | 0.00 | 0.00 | 0.44 |
| simple | 0.90 | 0.88 | 0.78 | 0.99 | 0.91 |
| agent | 0.90 | 0.88 | 0.78 | 0.99 | 0.77 |

**LLM-as-judge** (agent, 50 examples, cached rubric scores): mean overall **4.79 / 5** (groundedness 4.49, safety 4.0, escalation_fit 4.7, brand_tone 3.5).

**Judge–human agreement** (40 items): Spearman **0.53**, exact agreement **0.85**, within-1 **1.0**, MAE **0.16**.

### Reading the table
Intent/escalation for **agent** and **simple** match in the offline path because both share the keyword classifier + escalation policy. The agent’s differentiator offline is **draft behavior** (structured, intent-aware rewrite). Live Groq mode upgrades classify+draft; cached offline path keeps graders unblocked without API quota.

Escalate **recall 0.99** is intentional: false auto-handles on safety/suspension are worse than extra human reviews.

## 4. Failure analysis — top 5 modes

### F1. Multi-signal / colliding keywords
**Example:** “Kidnap? … I got a troll who suggested kidnapping”  
Gold: `spam_or_bots` · Pred: `thanks_or_other` (then over-escalated via low retrieval confidence).  
**Hypothesis:** Threat-adjacent slang without abuse lexicon terms falls through to vague-help; need better safety phrase coverage and multi-label support.

### F2. Vague help with feature-ish tokens
**Example:** “@TwitterSupport can I get some help please?”  
Gold: `thanks_or_other` · Pred: `feature_how_to` (triggered by “can I”).  
**Hypothesis:** Function words in how-to patterns are too broad; require stronger feature verbs (mute/block/pin) before assigning `feature_how_to`.

### F3. Emotional / pile-on messages without issue text
**Example:** “soul crushing @TwitterSupport COME ON LADS HELP US OUT”  
Gold: no escalate · Pred: escalate (`low_retrieval_sim`).  
**Hypothesis:** Retrieval confidence is a blunt instrument for short emotional tweets; should gate on length + intent risk, not similarity alone.

### F4. Typos and adversarial spelling
**Example (earlier sweep):** “harrass” misspelling missed `harass` until we added the variant.  
**Hypothesis:** Real abuse reports are adversarially misspelled; keyword systems need variant lists or fuzzy match.

### F5. Template reply “wins” embedding similarity without helping
Nearest-neighbor copies often score **embedding similarity = 1.0** against other templated brand replies — even when the customer’s issue was underspecified.  
**Hypothesis:** Reply embedding similarity rewards template reuse, not helpfulness; judge/human scores are the better draft metric.

## 5. What is misleading about my headline number?

**Intent accuracy 0.90 looks strong — and it is partly circular.**

Golden labels were produced with a **documented priority-keyword rubric** that is deliberately close to the offline keyword classifier. That makes the number a good regression test for routing consistency, but a **weak estimate of true human agreement** on ambiguous tweets (the modal TwitterSupport inbound is “help”).

Other caveats:
- **Escalate F1** privileges recall; precision ~0.65 means many extra escalations (cost of humans, not safety).
- **Reply emb-sim** favors the simple baseline’s raw copy (0.91) over the agent’s adapted draft (0.77) — higher is not better here.
- **Judge overall ~4.8** in the default cache uses a heuristic judge when Groq is unavailable; treat live Groq judge runs as the stronger quality signal.
- Real TwitterSupport volume is only **~1.3k** pairs; synthetic augmentation helps retrieval coverage but can smooth over brand-specific quirks.

## 6. What I’d do with one more week

1. True double-blind human labels on 200 items (replace rubric proximity).
2. Groq live sweep on full golden set + calibrate judge vs humans (target Spearman ≥ 0.7).
3. Multi-label intents + a dedicated `vague_help` class separate from thanks.
4. Learning-to-escalate: lightweight classifier on (intent, sim, length, toxicity proxy).
5. Error-driven keyword/lexicon expansion from the failure JSONL.
6. Tiny LoRA or logistic head on embeddings for intent (still offline-capable).

## 7. Data & labelling note

See [data/golden/README.md](data/golden/README.md). Summary: stratified sample of **real** TwitterSupport pairs, hard-case oversampling, priority-keyword labels + escalation policy, optional interactive review CLI.
