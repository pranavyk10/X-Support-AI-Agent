# Citations / borrowing

- **Dataset:** Thought Vector et al., [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), Kaggle.
- **Sentence embeddings:** Reimers & Gurevych, *Sentence-BERT*, model `sentence-transformers/all-MiniLM-L6-v2` via the `sentence-transformers` library.
- **LLM hosting:** Groq API — `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`.
- **Eval ideas:** LLM-as-judge rubrics inspired by common practice in MT-Bench / Chatbot Arena-style evaluation (Zheng et al.); implemented here as a fixed 1–5 multi-dimension rubric, not pairwise battles.
- **Download helper:** `kagglehub` for dataset fetch.
- No substantial third-party agent code was copied; prompts and escalation policy are original to this repo.
