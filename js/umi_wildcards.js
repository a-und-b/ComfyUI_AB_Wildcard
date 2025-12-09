import { app } from "../../scripts/app.js";

// =============================================================================
// PART 1: AUTOCOMPLETE LOGIC (Enhanced with Fuzzy Search)
// =============================================================================

class AutoCompletePopup {
    constructor() {
        this.element = document.createElement("div");
        Object.assign(this.element.style, {
            position: "absolute",
            display: "none",
            backgroundColor: "#1e1e1e",
            border: "1px solid #61afef", 
            zIndex: "9999",
            maxHeight: "250px",
            overflowY: "auto",
            color: "#e0e0e0",
            fontFamily: "'Consolas', 'Monaco', monospace",
            fontSize: "13px",
            borderRadius: "4px",
            boxShadow: "0 10px 25px rgba(0,0,0,0.8)",
            minWidth: "250px"
        });
        document.body.appendChild(this.element);
        
        this.visible = false;
        this.items = [];
        this.selectedIndex = 0;
        this.onSelectCallback = null;
    }

    show(x, y, options, onSelect) {
        this.items = options;
        this.onSelectCallback = onSelect;
        this.selectedIndex = 0; 
        this.visible = true;

        this.element.style.left = x + "px";
        this.element.style.top = y + "px";
        this.element.style.display = "block";
        this.render();
    }

    hide() {
        this.element.style.display = "none";
        this.visible = false;
        this.items = [];
    }

    render() {
        this.element.innerHTML = "";

        // Header
        const header = document.createElement("div");
        Object.assign(header.style, {
            padding: "4px 8px", fontSize: "11px", color: "#888",
            borderBottom: "1px solid #333", backgroundColor: "#252525"
        });
        header.innerText = this.items.length > 50 
            ? `Showing 50 of ${this.items.length} matches...` 
            : `${this.items.length} Suggestions`;
        this.element.appendChild(header);

        // List Items (Limit to 50 for performance)
        this.items.slice(0, 50).forEach((opt, index) => {
            const div = document.createElement("div");
            div.innerText = opt;
            
            Object.assign(div.style, {
                cursor: "pointer", padding: "6px 10px",
                borderBottom: "1px solid #2a2a2a", transition: "background 0.05s"
            });

            if (index === this.selectedIndex) {
                div.style.backgroundColor = "#2d4f6c"; 
                div.style.color = "#fff";
                div.style.borderLeft = "3px solid #61afef";
            } else {
                div.style.backgroundColor = "transparent";
                div.style.borderLeft = "3px solid transparent";
            }
            
            div.onmouseover = () => {
                this.selectedIndex = index;
                this.render(); 
            };
            
            div.onmousedown = (e) => {
                e.preventDefault(); 
                this.triggerSelection();
            };
            
            this.element.appendChild(div);
        });

        // Auto-Scroll
        if (this.element.children[this.selectedIndex + 1]) {
            const activeEl = this.element.children[this.selectedIndex + 1];
            if (activeEl.offsetTop < this.element.scrollTop) {
                this.element.scrollTop = activeEl.offsetTop;
            } else if (activeEl.offsetTop + activeEl.offsetHeight > this.element.scrollTop + this.element.offsetHeight) {
                this.element.scrollTop = activeEl.offsetTop + activeEl.offsetHeight - this.element.offsetHeight;
            }
        }
    }

    navigate(direction) {
        if (!this.visible) return;
        const max = Math.min(this.items.length, 50) - 1;
        if (direction === 1) {
            this.selectedIndex = this.selectedIndex >= max ? 0 : this.selectedIndex + 1;
        } else {
            this.selectedIndex = this.selectedIndex <= 0 ? max : this.selectedIndex - 1;
        }
        this.render();
    }

    triggerSelection() {
        if (this.visible && this.items[this.selectedIndex] && this.onSelectCallback) {
            this.onSelectCallback(this.items[this.selectedIndex]);
            this.hide();
        }
    }
}

// =============================================================================
// PART 2: THE PROFESSIONAL USER GUIDE UI
// =============================================================================

const HELP_STYLES = `
    .umi-help-modal {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.85); z-index: 10000;
        display: flex; justify-content: center; align-items: center;
        backdrop-filter: blur(8px); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .umi-help-content {
        background: #181818; width: 1000px; max-width: 95%; height: 90%;
        border-radius: 12px; box-shadow: 0 0 60px rgba(0,0,0,0.9);
        border: 1px solid #333; display: flex; flex-direction: column; overflow: hidden;
    }
    .umi-help-header {
        background: #202020; padding: 20px 40px; border-bottom: 1px solid #333;
        display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
    }
    .umi-help-header h2 { margin: 0; color: #fff; font-size: 24px; font-weight: 600; letter-spacing: 0.5px; }
    .umi-help-header .version { font-size: 12px; color: #666; font-weight: normal; margin-left: 10px; background: #111; padding: 2px 6px; border-radius: 4px; }
    .umi-help-close {
        background: #333; color: #ccc; border: 1px solid #444; padding: 8px 20px;
        border-radius: 6px; cursor: pointer; font-weight: 600; transition: 0.2s;
    }
    .umi-help-close:hover { background: #cc4444; color: white; border-color: #cc4444; }
    .umi-help-body {
        padding: 40px; overflow-y: auto; color: #ccc; line-height: 1.7;
        scrollbar-width: thin; scrollbar-color: #444 #181818;
    }
    
    /* Layout */
    .umi-section { margin-bottom: 50px; }
    .umi-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 20px; }
    
    /* Typography */
    h3 { color: #61afef; border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 0; font-size: 20px; font-weight: 600; display: flex; align-items: center; }
    h4 { color: #e5c07b; margin-bottom: 8px; margin-top: 20px; font-size: 15px; font-weight: 600; }
    p { margin-top: 0; font-size: 14px; color: #abb2bf; }
    
    /* Components */
    .umi-code {
        background: #282c34; padding: 2px 6px; border-radius: 4px; 
        font-family: "Consolas", "Monaco", monospace; color: #98c379; border: 1px solid #3e4451; font-size: 0.9em;
    }
    .umi-block {
        background: #282c34; padding: 15px; border-radius: 6px; 
        font-family: "Consolas", "Monaco", monospace; color: #abb2bf; border-left: 4px solid #61afef;
        margin: 10px 0; white-space: pre-wrap; font-size: 12px; overflow-x: auto;
    }
    .umi-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 15px; border: 1px solid #333; }
    .umi-table th { text-align: left; background: #252525; border-bottom: 1px solid #444; padding: 10px; color: #fff; }
    .umi-table td { border-bottom: 1px solid #333; padding: 10px; color: #bbb; background: #1e1e1e; }
    .umi-table tr:last-child td { border-bottom: none; }
    
    /* Callouts */
    .callout { padding: 15px; border-radius: 6px; margin-top: 20px; font-size: 13px; border-left: 4px solid; }
    .callout-info { background: #1c242c; border-color: #61afef; color: #d1d9e6; }
    .callout-warn { background: #2c2222; border-color: #e06c75; color: #e6d1d1; }
    .callout-success { background: #1e2620; border-color: #98c379; color: #d1e6d6; }
    
    /* Wiring Diagram Style */
    .step-list { margin: 0; padding: 0; list-style: none; counter-reset: step; }
    .step-list li { position: relative; padding-left: 30px; margin-bottom: 10px; font-size: 14px; }
    .step-list li::before { 
        counter-increment: step; content: counter(step); 
        position: absolute; left: 0; top: 0; width: 20px; height: 20px; 
        background: #333; color: #fff; border-radius: 50%; 
        text-align: center; line-height: 20px; font-size: 11px; font-weight: bold;
    }

    /* Details/Summary */
    details { background: #21252b; border-radius: 6px; padding: 10px; margin-bottom: 10px; border: 1px solid #333; transition: 0.2s; }
    details[open] { background: #282c34; border-color: #444; }
    summary { cursor: pointer; font-weight: 600; color: #e0e0e0; outline: none; list-style: none; display: flex; justify-content: space-between; align-items: center; }
    summary::after { content: "+"; color: #61afef; font-weight: bold; font-size: 16px; }
    details[open] summary::after { content: "−"; }
    details[open] summary { margin-bottom: 15px; border-bottom: 1px solid #3e4451; padding-bottom: 10px; }
`;

const HELP_HTML = `
    <div class="umi-section">
        <h3>🔌 Setup & Wiring (Lightweight)</h3>
        <p>Powerful wildcard text processor for ComfyUI prompts.</p>
        
        <div class="umi-grid-2">
            <div class="callout callout-success" style="margin-top: 0;">
                <h4 style="margin-top:0">Wiring</h4>
                <ul class="step-list">
                    <li>This node outputs processed <strong>String (Text)</strong> for your prompts.</li>
                    <li>Connect <strong>Text Output</strong> &#10142; <strong>CLIP Text Encode</strong>.</li>
                    <li>Use with any ComfyUI workflow that accepts text input.</li>
                </ul>
            </div>
            
            <div class="callout callout-info" style="margin-top: 0;">
                <h4 style="margin-top:0">Wildcards & Logic</h4>
                <ul class="step-list">
                    <li>Use <strong>Wildcards</strong> to randomize prompts.</li>
                    <li>Use <strong>Logic Gates</strong> to conditionally change text.</li>
                    <li><strong>Hot-Swap:</strong> Click <strong>"Refresh Wildcards"</strong> to reload files without restarting.</li>
                </ul>
            </div>
        </div>
        
         <div class="callout callout-warn">
            <strong>⚠️ Note on Batch Size:</strong><br>
            Use the <strong>"Queue Batch"</strong> setting in the ComfyUI Extra Options menu to generate variations. Do not use the widget batch size on the Latent node.
        </div>
    </div>

    <div class="umi-section">
        <h3>⚡ Syntax Cheat Sheet</h3>
        <div class="umi-grid-2">
            <div>
                <h4 style="margin-top:0">🎲 Dynamic Prompts</h4>
                <table class="umi-table">
                    <tr><td><span class="umi-code">{a|b}</span></td><td>Random choice.</td></tr>
                    <tr><td><span class="umi-code">{5::a|3::b|c}</span></td><td><strong>Weighted:</strong> a=5, b=3, c=1 probability.</td></tr>
                    <tr><td><span class="umi-code">{2$$a|b|c}</span></td><td>Pick 2 unique.</td></tr>
                    <tr><td><span class="umi-code">{2$$ and $$a|b}</span></td><td>Pick 2, join with " and ".</td></tr>
                    <tr><td><span class="umi-code">{a|{b|c}}</span></td><td><strong>Nesting:</strong> Unlimited depth.</td></tr>
                    <tr><td><span class="umi-code">__*__</span></td><td><strong>Wildcard:</strong> Pick from ANY file.</td></tr>
                    <tr><td><span class="umi-code">&lt;[Tag]&gt;</span></td><td><strong>Tag Aggregation:</strong> Pick from any entry with this tag.</td></tr>
                </table>
            </div>
            <div>
                <h4 style="margin-top:0">🎛️ Tools & Logic</h4>
                <table class="umi-table">
                    <tr><td><span class="umi-code">$var={...}</span></td><td>Define variable.</td></tr>
                    <tr><td><span class="umi-code">[if K : A | B]</span></td><td>Logic Gate.</td></tr>
                    <tr><td><span class="umi-code">[shuffle: a, b]</span></td><td>Randomize order.</td></tr>
                    <tr><td><span class="umi-code">[clean: a, , b]</span></td><td>Fix bad formatting.</td></tr>
                </table>
            </div>
        </div>
    </div>
    
    <div class="umi-section">
        <h3>📂 Creating Wildcards</h3>
        <p>You can create your own lists in the <code>wildcards/</code> folder.</p>
        
        <div class="umi-grid-2">
            <div>
                <h4>1. Simple Text Lists (.txt)</h4>
                <p>Create a file named <code>colors.txt</code>:</p>
                <div class="umi-block">Red
Blue
Green</div>
                <p><strong>Usage:</strong></p>
                <div class="umi-block">A __colors__ dress.</div>
            </div>
            
            <div>
                <h4>2. Umi YAML Format (Tag Aggregation)</h4>
                <div class="umi-block">Silk:
  Prompts: ["white hair archer"]
  Tags: [Demihuman]</div>
                <p style="font-size:12px">Usage: <code>&lt;[Demihuman]&gt;</code> (Picks any entry with 'Demihuman' tag) or <code>&lt;[Silk]&gt;</code>.</p>
            </div>
        </div>
    </div>
`;

function showHelpModal() {
    if (!document.getElementById("umi-help-style")) {
        const style = document.createElement("style");
        style.id = "umi-help-style";
        style.innerHTML = HELP_STYLES;
        document.head.appendChild(style);
    }

    const modal = document.createElement("div");
    modal.className = "umi-help-modal";
    modal.onclick = (e) => { if(e.target === modal) modal.remove(); };

    modal.innerHTML = `
        <div class="umi-help-content">
            <div class="umi-help-header">
                <div>
                    <h2>📘 AB Wildcard Reference Manual</h2>
                </div>
                <button class="umi-help-close" onclick="this.closest('.umi-help-modal').remove()">CLOSE</button>
            </div>
            <div class="umi-help-body">
                ${HELP_HTML}
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

// =============================================================================
// PART 3: REGISTRATION & FILTERING
// =============================================================================

// Helper: Custom fuzzy search function for client-side filtering
function getFuzzyMatches(query, allItems) {
    if (!query || query.trim() === "") {
        return allItems.sort(); 
    }
    
    const lowerQuery = query.toLowerCase();
    
    const scored = allItems.map(item => {
        const lowerItem = item.toLowerCase();
        
        if (lowerItem === lowerQuery) return { item, score: 100 };
        if (lowerItem.startsWith(lowerQuery)) return { item, score: 75 };
        if (lowerItem.includes(lowerQuery)) return { item, score: 50 };
        
        let qIdx = 0;
        let fuzzyScore = 0;
        for (let i = 0; i < lowerItem.length; i++) {
            if (lowerItem[i] === lowerQuery[qIdx]) {
                qIdx++;
                fuzzyScore += (100 - i); 
            }
            if (qIdx === lowerQuery.length) break;
        }
        
        if (qIdx === lowerQuery.length) {
            return { item, score: 10 + (fuzzyScore / 100) };
        }
        
        return { item, score: 0 };
    });

    return scored
        .filter(s => s.score > 0)
        .sort((a, b) => b.score - a.score)
        .map(s => s.item);
}

app.registerExtension({
    name: "ABWildcard.System",
    async setup() {
        this.wildcards = [];

        // Define a function we can call later to refresh the lists
        this.fetchWildcards = async () => {
             try {
                const resp = await fetch("/ab_wildcard/wildcards");
                if (resp.ok) {
                    const data = await resp.json();
                    
                    if (Array.isArray(data)) {
                        this.wildcards = data;
                    } else {
                        this.wildcards = data.wildcards || [];
                    }
                } else {
                    this.wildcards = [];
                }
            } catch (e) {
                console.error("[UmiAI] Failed to load wildcards:", e);
                this.wildcards = [];
            }
        };

        // Initial fetch
        await this.fetchWildcards();
        this.popup = new AutoCompletePopup();
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ABWildcardNode") return;

        // 1. Add Help Menu
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) getExtraMenuOptions.apply(this, arguments);
            options.push(null);
            options.push({
                content: "📘 Open AB Wildcard Guide",
                callback: () => { showHelpModal(); }
            });
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            const self = this;

            // ============================================================
            // NEW: REFRESH BUTTON WIDGET (WITH TOAST)
            // ============================================================
            this.addWidget("button", "🔄 Refresh Wildcards", null, () => {
                const btn = self.widgets.find(w => w.name.includes("Refresh"));
                
                // Visual feedback: Loading on Button (as fallback)
                if(btn) {
                    btn.name = "⏳ Refreshing...";
                    app.canvas.setDirty(true);
                }
                
                fetch("/ab_wildcard/refresh", { method: "POST" })
                    .then(r => r.json())
                    .then(data => {
                        // Re-fetch the autocomplete list in JS
                        const ext = app.extensions.find(e => e.name === "ABWildcard.System");
                        if(ext && ext.fetchWildcards) ext.fetchWildcards();
                        
                        // Use Modern Toast if available
                        if (app.extensionManager && app.extensionManager.toast) {
                            app.extensionManager.toast.add({
                                severity: "success",
                                summary: "Wildcards Refreshed",
                                detail: `Found ${data.count} wildcards.`,
                                life: 3000
                            });
                        }

                        // Reset button
                        if(btn) {
                            btn.name = "🔄 Refresh Wildcards"; 
                            app.canvas.setDirty(true);
                        }
                    })
                    .catch(e => {
                        console.error("[UmiAI] Refresh failed", e);
                         if (app.extensionManager && app.extensionManager.toast) {
                            app.extensionManager.toast.add({
                                severity: "error",
                                summary: "Refresh Failed",
                                detail: e.toString(),
                                life: 5000
                            });
                        }
                        if(btn) {
                            btn.name = "❌ Error";
                            app.canvas.setDirty(true);
                        }
                    });
            });

            // ============================================================
            // AUTOCOMPLETE LOGIC (WITH ARROW KEYS & FUZZY SEARCH)
            // ============================================================
            const textWidget = this.widgets.find(w => w.name === "text");
            if (!textWidget || !textWidget.inputEl) return;

            const inputEl = textWidget.inputEl;
            const ext = app.extensions.find(e => e.name === "ABWildcard.System");

            // 1. INTERCEPT NAVIGATION (Arrow Keys, Enter, Tab)
            inputEl.addEventListener("keydown", (e) => {
                if (ext.popup.visible) {
                    if (e.key === "ArrowDown") {
                        e.preventDefault();
                        ext.popup.navigate(1); // Next
                        return;
                    } 
                    if (e.key === "ArrowUp") {
                        e.preventDefault();
                        ext.popup.navigate(-1); // Prev
                        return;
                    }
                    if (e.key === "Enter" || e.key === "Tab") {
                        e.preventDefault();
                        ext.popup.triggerSelection();
                        return;
                    }
                    if (e.key === "Escape") {
                        ext.popup.hide();
                        return;
                    }
                }
            });

            // 2. LISTEN FOR TYPING (To show the popup)
            inputEl.addEventListener("keyup", (e) => {
                // Ignore nav keys in this listener to prevent flashing
                if (["ArrowUp", "ArrowDown", "Enter", "Escape"].includes(e.key)) return;

                const cursor = inputEl.selectionStart;
                const text = inputEl.value;
                const beforeCursor = text.substring(0, cursor);

                // Regex Config
                // Group 1: Opener (__ or <[)
                // Group 2: The query string
                const matchFile = beforeCursor.match(/(__|<\[)([\w\/\-\s]*)$/); 

                if (!ext) return;

                let options = [];
                let triggerType = ""; 
                let matchIndex = 0;
                let query = "";
                let opener = "";

                // -- Wildcard Logic --
                if (matchFile) {
                    triggerType = "file";
                    opener = matchFile[1]; // Capture the opener (__ or <[)
                    query = matchFile[2]; 
                    matchIndex = matchFile.index;
                    
                    // FUZZY SEARCH IMPLEMENTATION
                    options = getFuzzyMatches(query, ext.wildcards);
                } 

                if (triggerType && options.length > 0) {
                    const rect = inputEl.getBoundingClientRect();
                    const topOffset = rect.top + 20 + (rect.height / 2); // Approximate pos
                    
                    ext.popup.show(rect.left + 20, topOffset, options, (selected) => {
                        let completion = "";
                        
                        // Smart Completion based on Opener
                        if (triggerType === "file") {
                            // If started with <[, close with ]>
                            if (opener === "<[") {
                                completion = `<[${selected}]>`;
                            } else {
                                completion = `__${selected}__`;
                            }
                        }

                        const prefix = text.substring(0, matchIndex);
                        const suffix = text.substring(cursor);
                        
                        inputEl.value = prefix + completion + suffix;
                        
                        // Notify ComfyUI of change
                        if(textWidget.callback) textWidget.callback(inputEl.value);
                        
                        // Move cursor to end of inserted tag
                        const newCursorPos = (prefix + completion).length;
                        inputEl.setSelectionRange(newCursorPos, newCursorPos);
                        inputEl.focus();
                    });
                } else {
                    ext.popup.hide();
                }
            });

            // Close on outside click
            document.addEventListener("mousedown", (e) => {
                if (ext && ext.popup && e.target !== ext.popup.element && !ext.popup.element.contains(e.target) && e.target !== inputEl) {
                    ext.popup.hide();
                }
            });
        };
    }
});
