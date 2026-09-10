"""Project path helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
JUDGE_DIR = DATA_DIR / "judge_agreement"
INDEX_DIR = DATA_DIR / "retrieval_index"
RESULTS_DIR = ROOT / "results"
INTENTS_PATH = ROOT / "src" / "intents" / "taxonomy.yaml"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, GOLDEN_DIR, JUDGE_DIR, INDEX_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)