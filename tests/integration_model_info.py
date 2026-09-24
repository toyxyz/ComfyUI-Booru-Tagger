"""Check the loader details API against a running headless ComfyUI server."""

import argparse
import json
import pathlib
import urllib.error
import urllib.parse
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:8199")
args = parser.parse_args()
base = args.base.rstrip("/")
config = json.loads((pathlib.Path(__file__).resolve().parents[1] / "models.json").read_text(encoding="utf-8"))

for model in config["model_url"]:
    url = base + "/booru-tagger/model-info?" + urllib.parse.urlencode({"model_name": model})
    with urllib.request.urlopen(url, timeout=10) as response:
        info = json.load(response)
    assert info["tag_count"] == config["tag_count"][model], model
    assert info["input_size"] == config["input_size"][model], model
    assert info["threshold"] == config["threshold"][model], model
    assert info["category_thresholds"] == config["category_thresholds"][model], model

with urllib.request.urlopen(base + "/extensions/ComfyUI-Booru-Tagger/model_info.js", timeout=10) as response:
    script = response.read().decode("utf-8")
assert "BooruTagger.ModelInfo" in script

try:
    urllib.request.urlopen(base + "/booru-tagger/model-info?model_name=unknown", timeout=10)
except urllib.error.HTTPError as error:
    assert error.code == 404
else:
    raise AssertionError("Unknown model was accepted")

print(f"Model details API: {len(config['model_url'])} models, frontend script, and 404 check OK")
