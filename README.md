# X Support AI Agent (Hiver SDE Intern Take-Home)

AI support agent for **TwitterSupport** (historical handle for X Support) built on the Kaggle *Customer Support on Twitter* dataset.

Given an inbound customer tweet, the agent:
1. **Classifies** intent (11 data-derived classes)
2. **Drafts** a reply grounded in historical brand responses (RAG)
3. **Decides** auto-handle vs escalate, with a stated reason

The proof (golden set + eval harness + failure analysis) is the point of this repo.

## Reproduce headline results (< 15 minutes)

```bash
# From repo root
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# Prints the same numbers as results/headline_metrics.json (no API key required)
python -m src.eval.run_eval --use-cache
```

Optional live Groq run (free tier):

```bash
copy .env.example .env   # then set GROQ_API_KEY
python -m src.eval.run_eval --live --judge-limit 40
```

Demo one message (offline-safe):

```bash
python -m src.agent.pipeline --text "@TwitterSupport my account was hacked" --classify-mode keyword --draft-mode structured
```

## Headline results (golden n=200)

| system | intent acc | intent macro-F1 | escalate F1 | escalate recall |
| --- | ---: | ---: | ---: | ---: |
| trivial | 0.39 | 0.06 | 0.00 | 0.00 |
| simple (keyword + NN reply) | 0.90 | 0.88 | 0.78 | 0.99 |
| **agent** (keyword + structured grounded draft) | **0.90** | **0.88** | **0.78** | **0.99** |

Agent vs simple differ mainly on **drafts** (structured rewrite vs raw nearest-neighbor copy). Live mode (`--live`) swaps classify/draft to Groq (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`).

LLM-as-judge (cached heuristic / optional Groq) mean overall ≈ **4.8 / 5**. Judge–human agreement on 40 items: Spearman ≈ **0.53**, exact ≈ **0.85**, within-1 ≈ **1.0**.

See [REPORT.md](REPORT.md) for framing, failure modes, and the mandatory “what is misleading about my headline number?” section. See [DECISIONS.md](DECISIONS.md) for the decision log.

## Dataset

- Primary: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (thoughtvector)
- Brand filter: `author_id == TwitterSupport` → **1,257** real customer→brand pairs
- Retrieval index also includes a clearly marked **synthetic augmentation** (2,500 rows) because real TwitterSupport volume is thin vs Amazon/Apple
- Golden set is sampled from **real** pairs only

Rebuild processed data (needs ~170MB download once):

```bash
python -m src.data.prepare --download --max-pairs 12000
# then optionally merge synthetic:
python -m src.data.synthetic --n 2500
python -m src.agent.retrieve --build --max-rows 3500
python -m src.data.golden --build --n 200
```

## Project layout

```
src/agent/          classify, retrieve, draft, escalate, pipeline
src/baselines/      trivial + retrieval-only
src/eval/           metrics, LLM judge, run_eval
src/data/           prepare, golden labelling, synthetic fallback
src/intents/taxonomy.yaml
data/golden/        200 labelled eval examples + sampling note
data/processed/     committed subsample (real + synthetic)
data/retrieval_index/  embeddings + meta (for offline retrieve)
results/            headline_metrics.json + cached outputs
```

## Submit

1. Push this repo (public, or private with access granted)
2. Submit via the [Hiver form](https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f) with repo link + report

## Citations

- Dataset: Thought Vector, *Customer Support on Twitter*, Kaggle
- Embeddings: Reimers & Gurevych, Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`)
- LLMs (optional live): Groq-hosted Llama 3.1 8B Instant & Llama 3.3 70B Versatile
- Evaluation ideas adapted from common RAG + LLM-as-judge practice (Zheng et al., MT-Bench / Chatbot Arena style pairwise judging — here used as a structured rubric)
