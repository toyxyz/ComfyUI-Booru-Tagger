import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const number = (value) => Number(value).toLocaleString("en-US");

function describe(info) {
    const groups = info.category_counts
        ? Object.entries(info.category_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([name, count]) => `${name} ${number(count)}`)
            .join(" · ")
        : "Available after metadata download";
    const repository = new URL(info.repository).pathname.slice(1);
    const thresholds = Object.entries(info.category_thresholds)
        .map(([name, value]) => `${name[0].toUpperCase()}${name.slice(1)} ${value}`)
        .join(" · ");
    return [
        `Tags: ${number(info.tag_count)}`,
        `Groups: ${groups}`,
        `Input: ${info.input_size} × ${info.input_size}  |  ${info.format}`,
        `Recommended: ${thresholds}`,
        `Node inputs: ${info.threshold} general / ${info.character_threshold} character`,
        ...(info.category_thresholds.rating !== undefined
            ? ["Rating: cutoff applies when use_best_threshold is on"] : []),
        `Files: ${info.installed ? "Installed" : "Download on first use"}`,
        `Repository: ${repository}`,
    ].join("\n");
}

app.registerExtension({
    name: "BooruTagger.ModelInfo",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Load Booru Tagger") return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const selector = this.widgets?.find((widget) => widget.name === "model_name");
            if (!selector) return result;

            const display = document.createElement("textarea");
            display.readOnly = true;
            display.spellcheck = false;
            display.setAttribute("aria-label", "Selected model information");
            display.value = "Loading model details...";
            Object.assign(display.style, {
                width: "100%",
                height: "180px",
                boxSizing: "border-box",
                resize: "none",
                border: "1px solid #555",
                borderRadius: "6px",
                background: "#242424",
                color: "#ddd",
                padding: "8px",
                font: "12px/1.45 monospace",
            });
            const details = this.addDOMWidget("model_details", "MODEL_INFO", display, {
                serialize: false,
            });
            details.computeSize = () => [Math.max(this.size[0], 380), 190];
            this.setSize([Math.max(this.size[0], 400), Math.max(this.size[1], 425)]);

            let requestId = 0;
            const refresh = async () => {
                const modelName = selector.value;
                const current = ++requestId;
                display.value = "Loading model details...";
                try {
                    const response = await api.fetchApi(
                        `/booru-tagger/model-info?model_name=${encodeURIComponent(modelName)}`,
                    );
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const info = await response.json();
                    if (current === requestId) display.value = describe(info);
                } catch (error) {
                    if (current === requestId) display.value = `Model details unavailable: ${error.message}`;
                }
                app.graph?.setDirtyCanvas(true, false);
            };

            const originalCallback = selector.callback;
            selector.callback = function () {
                const callbackResult = originalCallback?.apply(this, arguments);
                refresh();
                return callbackResult;
            };
            const originalConfigure = this.onConfigure;
            this.onConfigure = function () {
                const configureResult = originalConfigure?.apply(this, arguments);
                queueMicrotask(refresh);
                return configureResult;
            };
            const originalExecuted = this.onExecuted;
            this.onExecuted = function () {
                const executedResult = originalExecuted?.apply(this, arguments);
                refresh();
                return executedResult;
            };
            const originalRemoved = this.onRemoved;
            this.onRemoved = function () {
                requestId++;
                return originalRemoved?.apply(this, arguments);
            };
            refresh();
            return result;
        };
    },
});
