from .nodes import ABWildcardNode
from server import PromptServer
from aiohttp import web
import os
import glob
import yaml

# 1. Setup the API Route
def get_wildcard_data():
    wildcards_path = os.path.join(os.path.dirname(__file__), "wildcards")
    files = []
    tags = set()
    
    if os.path.exists(wildcards_path):
        # 1. Recursive search for files
        for ext in ['*.txt', '*.yaml']:
            for filepath in glob.glob(os.path.join(wildcards_path, '**', ext), recursive=True):
                # Get relative path for __file__
                rel_path = os.path.relpath(filepath, wildcards_path)
                tag_name = os.path.splitext(rel_path)[0].replace(os.sep, '/')
                files.append(tag_name)

                # 2. If YAML, parse it for Tags
                if ext == '*.yaml':
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                            if isinstance(data, dict):
                                for entry in data.values():
                                    if isinstance(entry, dict) and 'Tags' in entry:
                                        # Add all tags to the set (deduplicated)
                                        for t in entry['Tags']:
                                            tags.add(str(t).strip())
                    except Exception as e:
                        print(f"[AB_Wildcard] Error parsing YAML {filepath}: {e}")

    # Return Files and Tags only
    return {
        "files": sorted(files),
        "tags": sorted(list(tags))
    }

# Register the route
@PromptServer.instance.routes.get("/ab_wildcard/wildcards")
async def fetch_wildcards(request):
    data = get_wildcard_data()
    return web.json_response(data)

# 2. Mappings
NODE_CLASS_MAPPINGS = {
    "ABWildcardNode": ABWildcardNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ABWildcardNode": "AB Wildcard"
}

# 3. Expose the web directory
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
