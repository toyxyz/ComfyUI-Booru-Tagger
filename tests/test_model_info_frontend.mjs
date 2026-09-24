import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

let extension;
const app = {
    registerExtension(value) { extension = value; },
    graph: { setDirtyCanvas() {} },
};
const responses = {
    "animetimm-convnextv2_huge-dbv4-full": {
        tag_count: 12476, input_size: 512, format: "ONNX",
        threshold: 0.38, character_threshold: 0.51,
        category_thresholds: { general: 0.38, character: 0.51, rating: 0.24 },
        repository: "https://huggingface.co/animetimm/convnextv2_huge.dbv4-full",
        installed: true, category_counts: { General: 9225, Character: 3247, Rating: 4 },
    },
    "wd-vit-tagger-v3": {
        tag_count: 10861, input_size: 448, format: "ONNX",
        threshold: 0.35, character_threshold: 0.85,
        category_thresholds: { general: 0.35, character: 0.85 },
        repository: "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3",
        installed: false, category_counts: null,
    },
};
const api = {
    async fetchApi(path) {
        const model = new URL(path, "http://localhost").searchParams.get("model_name");
        return { ok: true, async json() { return responses[model]; } };
    },
};
const document = {
    createElement() {
        return { style: {}, setAttribute() {} };
    },
};

const source = readFileSync(new URL("../web/model_info.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");
runInNewContext(source, { app, api, document, URL, Number, Object, encodeURIComponent, queueMicrotask });

const nodeType = { prototype: {} };
await extension.beforeRegisterNodeDef(nodeType, { name: "Load Booru Tagger" });
const selector = { name: "model_name", value: "animetimm-convnextv2_huge-dbv4-full" };
const node = {
    widgets: [selector], size: [400, 220],
    addDOMWidget(name, type, element, options) {
        assert.equal(name, "model_details");
        assert.equal(options.serialize, false);
        this.display = element;
        return {};
    },
    setSize(size) { this.size = size; },
};
nodeType.prototype.onNodeCreated.call(node);
await new Promise(setImmediate);
assert.match(node.display.value, /Tags: 12,476/);
assert.match(node.display.value, /General 9,225/);
assert.match(node.display.value, /Rating 0.24/);
assert.match(node.display.value, /cutoff applies when use_best_threshold is on/);
assert.match(node.display.value, /Files: Installed/);

selector.value = "wd-vit-tagger-v3";
selector.callback();
await new Promise(setImmediate);
assert.match(node.display.value, /Tags: 10,861/);
assert.match(node.display.value, /Download on first use/);
console.log("Loader details widget updates when the selected model changes");
