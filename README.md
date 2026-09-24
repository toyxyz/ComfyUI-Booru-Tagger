# ComfyUI Booru Tagger

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension that tags images with booru tags for use in prompts. It bundles the commonly used open tagger models — [WD](https://huggingface.co/SmilingWolf), [Pixai](https://huggingface.co/pixai-labs/pixai-tagger-v1.0), [Camie](https://huggingface.co/Camais03/camie-tagger-v2), [CL Tagger](https://huggingface.co/cella110n) and [AnimeTimm](https://huggingface.co/animetimm) — 24 taggers in total.

## Features

- **Model choice** — 24 taggers across WD, Pixai, Camie, CL Tagger and AnimeTimm; AnimeTimm models use 12,476-tag vocabularies. Use the one you prefer.
- **Model details in the loader** — selecting a model shows its tag count, input size, format, default thresholds, repository, and download status. Category counts appear when tag metadata is installed; model weights are not loaded just to show details.
- **Loads once** — a model is loaded the first time it is used and kept in memory, so later images do not wait for another load.
- **Multiple images per run** — the node accepts a batch of images and returns tags for all of them.
- **Tag groups** — results are split into general, rating and character (character/copyright/artist) tags, with options for underscores, sorting, tag exclusions and per-model thresholds.
- **Automatic downloads** — missing model files are downloaded on first use; gated models require a HuggingFace token (accepted once on the model page).

**use_best_threshold** defaults to enabled. Non-rating tags must meet their model's category recommendation, any available per-tag `best_threshold`, and the connected `threshold` or `character_threshold` input. The highest-scoring rating is returned only if it meets a published rating threshold. Disable this option to use only the two node inputs for non-rating tags and always return the top rating. AnimeTimm general / character recommendations are eva02 (`0.39` / `0.61`), caformer (`0.39` / `0.47`), swinv2 (`0.41` / `0.59`), and ConvNeXtV2 Huge (`0.38` / `0.51`).

Pixai Tagger v1.0 uses the official PyTorch model and downloads `model.safetensors`, `config.json`, and `tagger_pipeline.py` on first use. It needs `transformers`, `timm`, and `safetensors` in the ComfyUI Python environment. Its six official category thresholds are general 0.17, character 0.27, style 0.15, copyright 0.24, meta 0.17, and rating 0.41. Style and meta tags appear under `general_tags`; copyright tags appear under `character_tags`.

The Booru Tagger's `threshold` and `character_threshold` inputs default to `0`, so a new node uses each model's recommendations automatically when `use_best_threshold` is enabled. Set either input higher to impose an additional minimum. Existing saved workflows retain their stored input values; lower them to `0` to use recommendations without an extra floor. The loader still outputs model default values for workflows that explicitly connect them. Pixai v1.0 outputs `0.15` / `0.24` as floors so its style and copyright recommendations remain reachable; its general and character categories use `0.17` / `0.27` through the per-category best thresholds.

| Model family | General | Character | Other published category thresholds | Source |
|---|---:|---:|---|---|
| WD v3 and v1.4 v2 | 0.35 | 0.85 | Rating: highest score, no published cutoff | [WD Tagger demo](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) |
| WD v1.4 original (2 models) | 0.35 | — | No character vocabulary; rating uses highest score | [WD Tagger demo](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) |
| Pixai v0.9 | 0.30 | 0.85 | No rating vocabulary | [Model card](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) |
| Pixai v1.0 | 0.17 | 0.27 | Style 0.15; copyright 0.24; meta 0.17; rating 0.41 | [Model card](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) |
| Camie v2 (macro profile) | 0.492 | 0.492 | Copyright, artist, meta, year, rating: each 0.492 | [Model card](https://huggingface.co/Camais03/camie-tagger-v2) |
| CL Tagger v1 (all 3 revisions) | 0.55 | 0.60 | Meta/model 0.55; copyright/artist 0.60; rating/quality highest score | [Creator's demo](https://huggingface.co/spaces/cella110n/cl_tagger/blob/main/app.py) |
| CL Tagger v2 (both revisions) | 0.55 | 0.55 | Single 0.55 cutoff for copyright, meta, rating, quality | [Model card](https://huggingface.co/cella110n/cl_tagger_v2) |
| AnimeTimm SwinV2 | 0.41 | 0.59 | Rating 0.41 | [Model card](https://huggingface.co/animetimm/swinv2_base_window8_256.dbv4-full) |
| AnimeTimm EVA02 | 0.39 | 0.61 | Rating 0.38 | [Model card](https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full) |
| AnimeTimm CAFormer | 0.39 | 0.47 | Rating 0.39 | [Model card](https://huggingface.co/animetimm/caformer_b36.dbv4-full) |
| AnimeTimm ConvNeXtV2 Huge | 0.38 | 0.51 | Rating 0.24 | [Model card](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) |

The loader's model-details panel shows every documented category threshold. Style, meta, model, year, and quality tags with published category cutoffs are filtered by their own recommendation and placed in `general_tags`; copyright and artist tags go to `character_tags`. CL v1 quality tags have no numeric recommendation and remain excluded. The `rating` output selects the highest-scoring rating and is empty when that score is below the model's published rating threshold.

## Outputs

| Output | Description |
|---|---|
| `tags` | Combined character + general tags |
| `general_tags` | Descriptive tags (attributes, clothing, composition, etc.) |
| `rating` | Top rating tag when its published cutoff is met; otherwise empty |
| `character_tags` | Character, copyright, and artist tags |

## Models

| Model | Parameters | Tags | Input Size | License | Gated |
|---|---|---|---|---|---|
| WD Series (eva02, vit, swinv2, etc.) | varies | 6,549 / 9,083 / 10,861 by version | 448² | MIT | No |
| Pixai Tagger v0.9 | — | 13,461 | 448² | Apache-2.0 | No |
| Pixai Tagger v1.0 (PyTorch) | 486.3M | 30,877 | 1008² | Apache-2.0 | No |
| Camie Tagger v2 | — | 70,527 | 512² | ? | No |
| CL Tagger v1 (1.00 / 1.01 / 1.02) | — | 42,163 / 43,638 / 51,213 | 448² | Apache-2.0 | No |
| CL Tagger v2 (2.00 / 2.01a) | — | 106,536 / 108,036 | 384² | Custom | **Yes** |
| AnimeTimm swinv2_base | 99.7M | 12,476 | 256² | GPL-3.0 | **Yes** |
| AnimeTimm caformer_b36 | 134.0M | 12,476 | 384² | GPL-3.0 | **Yes** |
| AnimeTimm eva02_large | 316.8M | 12,476 | 448² | GPL-3.0 | **Yes** |
| AnimeTimm ConvNeXtV2 Huge (community ONNX) | 692.6M | 12,476 | 512² | GPL-3.0 | **Yes** |

> ConvNeXtV2 Huge uses the community ONNX conversion from [itterative](https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx), with official AnimeTimm metadata and preprocessing.

> The original ConvNeXtV2 Huge repository gates `preprocess.json`. When that file is absent, this extension uses the 512px white padding, bicubic resize, center crop, and ImageNet normalization published in the [model card](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full).

> **Gated models require a HuggingFace token.** Accept the license on the model page, then either run `huggingface-cli login` or set the `HF_TOKEN` environment variable before first download.

Credits:
- [pythongosssss/ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)
- [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags)
- [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)

Models created by:
- WD Taggers: [SmilingWolf](https://huggingface.co/SmilingWolf)
- Pixai Tagger: [pixai-labs](https://huggingface.co/pixai-labs)
- Camie Tagger: [Camais03](https://huggingface.co/Camais03)
- CL Tagger v1 / v2: [cella110n](https://huggingface.co/cella110n)
- AnimeTimm: [DeepGHS](https://huggingface.co/deepghs) / [narugo1992](https://huggingface.co/narugo1992)

## Installation

1. Install from the [Comfy Registry](https://registry.comfy.org/nodes/booru-tagger) (ComfyUI-Manager or `comfy node install booru-tagger`), or clone this repo into the `custom_nodes` folder.
2. Install dependency (`onnxruntime` or `onnxruntime-gpu` for CUDA acceleration). When installing via the registry this is handled automatically.
3. For gated models (CL Tagger v2, AnimeTimm): run `huggingface-cli login` once, or set the `HF_TOKEN` environment variable with your HuggingFace token.

## Configuration

Edit `models.json` to customize defaults:

- **`settings.ortProviders`** — ONNX Runtime execution providers. Providers that the installed `onnxruntime` build does not ship are dropped automatically (with a warning).
- **`settings.preprocess`** — `"tensor"` (default) resizes/pads/normalises on the GPU straight from ComfyUI's image tensor; `"pil"` restores the original CPU/PIL path (same tags, ~13 ms slower per image).
- **`threshold` / `character_threshold`** — Per-model default thresholds.
- **`HF_ENDPOINT`** — Mirror/proxy URL for HuggingFace downloads.
