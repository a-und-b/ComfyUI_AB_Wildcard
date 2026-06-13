from .nodes import ABWildcardNode

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ABWildcardNode": ABWildcardNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ABWildcardNode": "AB Wildcard"
}

# Expose the web directory (frontend autocomplete)
WEB_DIRECTORY = "./js"

# Note: the /ab_wildcard/wildcards and /ab_wildcard/refresh HTTP routes are
# registered in nodes.py, the single source of truth for the API.

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
