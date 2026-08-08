import json
from pathlib import Path


def load_config(project_root, method):
    path = Path(project_root) / "Method" / "Router" / "configs" / f"{method}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_text(value):
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return str(value or "").strip()


def load_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows
