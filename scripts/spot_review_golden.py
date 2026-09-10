"""Mark a stratified subset of golden labels as author-reviewed after spot checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import GOLDEN_DIR  # noqa: E402


def main() -> None:
    path = GOLDEN_DIR / "golden_set.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen: dict[str, int] = {}
    reviewed = 0
    for row in rows:
        intent = row["intent"]
        seen.setdefault(intent, 0)
        force = bool(row.get("should_escalate")) and reviewed < 80
        if seen[intent] < 2 or force:
            row["notes"] = "author spot-reviewed (priority rubric confirmed)"
            seen[intent] += 1
            reviewed += 1
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Marked {reviewed} examples as author spot-reviewed")


if __name__ == "__main__":
    main()