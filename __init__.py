from .nodes import ABWildcardNode
from .tools import ABWildcardLoad, ABWildcardSave

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ABWildcardNode": ABWildcardNode,
    "ABWildcardSave": ABWildcardSave,
    "ABWildcardLoad": ABWildcardLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ABWildcardNode": "AB Wildcard",
    "ABWildcardSave": "AB Wildcard Save",
    "ABWildcardLoad": "AB Wildcard Load",
}

# Expose the web directory (frontend autocomplete)
WEB_DIRECTORY = "./js"

# Note: the /ab_wildcard/wildcards and /ab_wildcard/refresh HTTP routes are
# registered in nodes.py, the single source of truth for the API.

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
