import json
import os
from typing import List, Dict, Any, Optional

DATASET_INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "label_dataset", "index.json"))


def _load_index() -> List[Dict[str, Any]]:
    if not os.path.exists(DATASET_INDEX):
        return []
    with open(DATASET_INDEX, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def load_label_samples(category: Optional[str] = None, difficulty: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
    """Return up to `limit` label entries filtered by category/difficulty.

    Entry schema in index.json:
    {
      "file_path": "absolute/or/relative/path.jpg",
      "category": "warehouse|grocery|apparel",
      "fields": {"product": "...", "weight": "..."},
      "difficulty": "easy|medium|hard",
      "language_hint": "en|hi",
      "region": "IN"
    }
    """
    items = _load_index()
    results: List[Dict[str, Any]] = []
    seen_fps = set()
    for item in items:
        if category and str(item.get("category", "")).lower() != category.lower():
            continue
        if difficulty and str(item.get("difficulty", "")).lower() != difficulty.lower():
            continue
        # Normalize file_paths: allow either file_path or file_paths
        if not item.get("file_paths") and item.get("file_path"):
            item = dict(item)
            item["file_paths"] = [item["file_path"]]
        # Deduplicate by tuple of file_paths to avoid repeated labels
        fps_key = tuple(item.get("file_paths") or [])
        if fps_key in seen_fps:
            continue
        seen_fps.add(fps_key)
        results.append(item)
        if len(results) >= max(1, limit):
            break
    return results


def read_label_fields(file_path: str) -> Dict[str, Any]:
    """Lookup fields for a given file_path from the index; returns fields dict.
    This is a metadata read (no OCR).
    """
    items = _load_index()
    for item in items:
        if os.path.normpath(str(item.get("file_path"))) == os.path.normpath(file_path):
            return item.get("fields", {})
    return {}


