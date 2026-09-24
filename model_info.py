"""Lightweight details for the selected model, without loading model weights."""

import csv
import json
import os
from collections import Counter
from functools import lru_cache

from .nodes import config, models_dir


_CATEGORY_NAMES = {"0": "General", "1": "Rating", "4": "Character", "9": "Rating"}


@lru_cache(maxsize=32)
def _read_metadata(path, mtime_ns, size):
    """Cache parsed local metadata and invalidate it when a download changes the file."""
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as stream:
            categories = Counter(row.get("category", "Other") for row in csv.DictReader(stream))
        count = sum(categories.values())
    else:
        with open(path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        if "tags" in metadata and "tags_split" in metadata:  # PixAI v1.0
            count = len(metadata["tags"])
            categories = Counter(dict(metadata["tags_split"]))
        else:
            mapping = metadata.get("dataset_info", {}).get("tag_mapping", metadata)
            if "idx_to_tag" in mapping:
                count = len(mapping["idx_to_tag"])
                categories = Counter(mapping.get("tag_to_category", {}).values())
            else:  # Alternate CL v1 tag mapping format.
                count = len(mapping)
                categories = Counter(item.get("category", "Other") for item in mapping.values())
    readable = Counter()
    for category, amount in categories.items():
        readable[_CATEGORY_NAMES.get(str(category), str(category).replace("_", " ").title())] += amount
    return count, dict(readable)


def get_model_info(model_name):
    """Return published details, with exact local metadata counts when available."""
    if model_name not in config["model_url"]:
        raise KeyError(model_name)

    model_path = os.path.join(models_dir, config["model_path"][model_name])
    metadata_path = os.path.join(models_dir, config["metadata_path"][model_name])
    required_paths = [model_path, metadata_path]
    for key in ("external_data_path", "code_path"):
        relative = config.get(key, {}).get(model_name)
        if relative:
            required_paths.append(os.path.join(models_dir, relative))
    preprocess = config.get("preprocess_path", {}).get(model_name)
    if preprocess and model_name not in config.get("optional_preprocess", []):
        required_paths.append(os.path.join(models_dir, preprocess))

    info = {
        "model_name": model_name,
        "tag_count": config["tag_count"][model_name],
        "input_size": config["input_size"][model_name],
        "format": "PyTorch" if model_path.endswith(".safetensors") else "ONNX",
        "threshold": config["threshold"][model_name],
        "character_threshold": config["character_threshold"][model_name],
        "category_thresholds": config["category_thresholds"][model_name],
        "repository": config["model_url"][model_name].replace(
            "{HF_ENDPOINT}", "https://huggingface.co"),
        "installed": all(os.path.isfile(path) for path in required_paths),
        "category_counts": None,
    }
    if os.path.isfile(metadata_path):
        stat = os.stat(metadata_path)
        try:
            info["tag_count"], info["category_counts"] = _read_metadata(
                metadata_path, stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError, KeyError, TypeError):
            # A partial or older metadata file should not hide the catalog details.
            pass
    return info
