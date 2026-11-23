import logging

from .nodes import ABWildcardNode

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ABWildcardNode": ABWildcardNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ABWildcardNode": "AB Wildcard",
}

# The helper nodes in tools.py use ComfyUI's V3 node API (display names live in their schema).
# On a ComfyUI without it, AB Wildcard itself must keep working.
try:
    from .tools import (ABLLMApply, ABLLMPrep, ABLoraLoader, ABPromptMix, ABTextLines, ABWildcardExtract,
                        ABWildcardLoad, ABWildcardSave)
except ImportError as e:
    logging.warning(f"[AB Wildcard] Save/Extract/Load/Text Lines/Prompt Mix/LLM Prep/Apply/LoRA Loader need a newer ComfyUI and are disabled: {e}")
else:
    for cls in (ABWildcardSave, ABWildcardExtract, ABWildcardLoad, ABTextLines, ABPromptMix, ABLLMPrep, ABLLMApply, ABLoraLoader):
        NODE_CLASS_MAPPINGS[cls.__name__] = cls

try:
    from .folder import ABImageFromFolder
except ImportError as e:
    logging.warning(f"[AB Wildcard] AB Image from Folder needs a newer ComfyUI and is disabled: {e}")
else:
    NODE_CLASS_MAPPINGS["ABImageFromFolder"] = ABImageFromFolder

# Expose the web directory (frontend autocomplete)
WEB_DIRECTORY = "./js"

# Note: the /ab_wildcard/wildcards and /ab_wildcard/refresh HTTP routes are
# registered in nodes.py, the single source of truth for the API.

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
