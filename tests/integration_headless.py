"""Run a real ComfyUI API workflow against an already running headless server."""

import argparse
import json
import time
import urllib.request
import uuid


def api(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def run_model(base, model, expected, override=None):
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
        "2": {"class_type": "Load Booru Tagger", "inputs": {
            "model_name": model, "replace_underscore": True}},
        "3": {"class_type": "Booru Tagger", "inputs": {
            "tagger_model": ["2", 0], "tagger_info": ["2", 1], "image": ["1", 0],
            "threshold": ["2", 2], "character_threshold": ["2", 3],
            "use_best_threshold": True, "trailing_comma": False,
            "sort_tags": False, "exclude_tags": ""}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["2", 2]}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["2", 3]}},
    }
    if override is not None:
        prompt["3"]["inputs"]["threshold"] = override[0]
        prompt["3"]["inputs"]["character_threshold"] = override[1]
    queued = api(base, "/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})
    if queued.get("node_errors"):
        raise AssertionError(f"{model}: prompt validation: {queued['node_errors']}")
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        history = api(base, "/history/" + prompt_id).get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise AssertionError(f"{model}: execution failed: {status}")
            outputs = history.get("outputs", {})
            if all(key in outputs for key in ("4", "5", "6")):
                values = [outputs[key]["text"][0] for key in ("5", "6")]
                actual = tuple(float(value) for value in values)
                if actual != expected:
                    raise AssertionError(f"{model}: thresholds {actual}, expected {expected}")
                tags = outputs["4"]["text"][0]
                if override == (1.0, 1.0) and tags:
                    raise AssertionError(f"{model}: node threshold override did not suppress tags")
                print(f"{model}: defaults={actual}, override={override}, inference OK, tag preview={tags[:120]!r}")
                return
        time.sleep(1)
    raise TimeoutError(f"{model}: no completed result in 180 seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8198")
    args = parser.parse_args()
    nodes = api(args.base, "/object_info")
    assert all(name in nodes for name in
               ("Load Booru Tagger", "Booru Tagger", "EmptyImage", "PreviewAny"))
    for model, expected in (
        ("wd-v1-4-moat-tagger-v2", (0.35, 0.85)),
        ("pixai-tagger-v0.9", (0.3, 0.85)),
        ("pixai-tagger-v1.0", (0.15, 0.24)),
        ("animetimm-convnextv2_huge-dbv4-full", (0.38, 0.51)),
    ):
        run_model(args.base, model, expected)
    run_model(args.base, "wd-v1-4-moat-tagger-v2", (0.35, 0.85), override=(1.0, 1.0))
