// AB Image from Folder: an always-visible thumbnail grid of the folder on the node.
// A click sets pick "selected" and the file widget; the grid marks the selected file,
// or the file the last run used. Routes: folder.py. Plan: docs/plan-image-from-folder.md
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "ABImageFromFolder";
const GRID_HEIGHT = 200;

const style = document.createElement("style");
style.textContent = `
.ab-thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 4px;
  align-content: start; overflow-y: auto; box-sizing: border-box; padding: 2px; height: ${GRID_HEIGHT}px;
  flex: none; /* Nodes 2.0 wraps DOM widgets in a flex column that would stretch it */ }
.ab-thumbs img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; cursor: pointer;
  outline: 3px solid transparent; outline-offset: -3px; background: rgba(127,127,127,.15); }
.ab-thumbs img:hover { outline-color: rgba(255,255,255,.5); }
.ab-thumbs img.ab-on { outline-color: #fff; box-shadow: 0 0 0 2px #4a9eff; }
.ab-thumbs .ab-msg { grid-column: 1 / -1; font-size: 11px; opacity: .7; padding: 4px; }`;
document.head.appendChild(style);

app.registerExtension({
    name: "a-und-b.ABWildcard.imageFolder",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            const w = (name) => node.widgets?.find((x) => x.name === name);

            const el = document.createElement("div");
            el.className = "ab-thumbs";
            // scroll the grid, not the canvas
            el.addEventListener("wheel", (e) => {
                if (el.scrollHeight > el.clientHeight) e.stopPropagation();
            }, { passive: true });
            el.addEventListener("click", (e) => {
                const img = e.target.closest("img[data-name]");
                if (!img) return;
                for (const [name, value] of [["pick", "selected"], ["file", img.dataset.name]]) {
                    const widget = w(name);
                    widget.value = value;
                    widget.callback?.(value, app.canvas, node);
                }
                node.setDirtyCanvas(true, true);
            });

            const grid = node.addDOMWidget("thumbs", "ab_thumbs", el, {
                serialize: false,  // not in the prompt…
                hideOnZoom: false,
                getMinHeight: () => GRID_HEIGHT + 8,
                getMaxHeight: () => GRID_HEIGHT + 8,
            });
            grid.serialize = false;  // …and not in widgets_values

            const message = (text) => {
                el.replaceChildren(Object.assign(document.createElement("div"), { className: "ab-msg", textContent: text }));
            };

            node.abMark = () => {
                const current = w("pick")?.value === "selected" ? w("file")?.value : node.abLast;
                for (const img of el.querySelectorAll("img")) img.classList.toggle("ab-on", img.dataset.name === current);
            };

            let timer, request = 0;
            node.abRefresh = () => {
                clearTimeout(timer);
                timer = setTimeout(async () => {
                    if (node.inputs?.find((i) => i.name === "folder")?.link != null) {
                        return message("folder is linked: no thumbnails before the run");
                    }
                    const params = { folder: w("folder")?.value ?? "", match: w("match")?.value ?? "", sort: w("sort")?.value ?? "" };
                    if (!params.folder.trim()) return message("Type a folder inside ComfyUI/input, or an absolute path");
                    const mine = ++request;
                    let data;
                    try {
                        data = await (await api.fetchApi("/ab_wildcard/folder/list?" + new URLSearchParams(params))).json();
                    } catch (e) {
                        data = { error: String(e) };
                    }
                    if (mine !== request) return;  // a newer refresh is on its way
                    if (data.error) return message(data.error);
                    el.replaceChildren(...data.files.map(([name, mtime]) => Object.assign(document.createElement("img"), {
                        loading: "lazy", title: name, draggable: false,
                        src: api.apiURL("/ab_wildcard/folder/thumb?" + new URLSearchParams({ ...params, name, v: mtime })),
                    })));
                    el.querySelectorAll("img").forEach((img, i) => (img.dataset.name = data.files[i][0]));
                    node.abMark();
                }, 300);
            };

            for (const name of ["folder", "match", "sort", "pick", "file"]) {
                const widget = w(name);
                if (!widget) continue;
                const callback = widget.callback;
                widget.callback = function () {
                    const result = callback?.apply(this, arguments);
                    ["pick", "file"].includes(name) ? node.abMark() : node.abRefresh();
                    return result;
                };
            }

            node.setSize([Math.max(node.size[0], 340), node.computeSize()[1]]);
            node.abRefresh();
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            this.abRefresh?.();
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);
            this.abRefresh?.();
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const r = onExecuted?.apply(this, arguments);
            this.abLast = output?.ab_file?.[0];
            this.abMark?.();
            return r;
        };
    },
});
