"""Test bootstrap.

nodes.py imports ComfyUI-only modules (`server`, `folder_paths`) at import
time and registers HTTP routes via decorators. We stub those so the pure text
processing engine can be imported and tested headlessly.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- stub: server (PromptServer route decorators) ---
if "server" not in sys.modules:
    server = types.ModuleType("server")

    class _Routes:
        def get(self, *a, **k):
            return lambda fn: fn

        def post(self, *a, **k):
            return lambda fn: fn

    class _Instance:
        routes = _Routes()

    class PromptServer:
        instance = _Instance()

    server.PromptServer = PromptServer
    sys.modules["server"] = server

# --- stub: folder_paths (point external roots at non-existent dirs so only the
# node's own bundled wildcards/ folder is used) ---
if "folder_paths" not in sys.modules:
    fp = types.ModuleType("folder_paths")
    fp.base_path = os.path.join(ROOT, "tests", "_no_root")
    fp.models_dir = os.path.join(ROOT, "tests", "_no_models")
    fp.get_folder_paths = lambda *_a, **_k: []
    sys.modules["folder_paths"] = fp
