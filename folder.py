"""AB Image from Folder: one image per run from a folder, picked by the seed.

Nothing is remembered between runs: the index comes from the seed alone, with the
same seed logic as {~ ! @} in AB Wildcard (SequenceContext). Same seed, same image,
so ComfyUI's cache, a restart and API runs all behave.
Plan: docs/plan-image-from-folder.md
"""
import fnmatch
import hashlib
import os
import random
import re

import folder_paths
from comfy_api.latest import io

from .nodes import SequenceContext

IN_ORDER, SHUFFLED, RANDOM = "in order", "shuffled, no repeats", "random"
BY_NAME, BY_DATE = "name", "date modified"
DEFAULT_MATCH = "*.png;*.jpg;*.jpeg;*.webp"
UINT64 = 0xffffffffffffffff
NAME = "AB Image from Folder"


# --- pure logic (pytest reaches it without ComfyUI) ---

def resolve_folder(folder, input_dir):
    """Absolute path for the folder widget. Relative paths stay inside input_dir."""
    folder = (folder or "").strip()
    if not folder:
        raise ValueError(f"{NAME}: no folder set. Type a folder inside ComfyUI/input, or an absolute path.")
    if os.path.isabs(folder):
        return os.path.normpath(folder)
    base = os.path.abspath(input_dir)
    path = os.path.abspath(os.path.join(base, folder))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"{NAME}: {folder!r} leaves the input folder ({base}). Use an absolute path instead.")
    return path


def _natural(name):
    # re.split with a group alternates text, number, text…: same types at the same positions
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_images(path, match=DEFAULT_MATCH, sort=BY_NAME):
    """Matching files in path (not recursive, no hidden files) as (name, path, mtime_ns, size)."""
    if not os.path.isdir(path):
        raise ValueError(f"{NAME}: folder not found: {path}")
    patterns = [p.strip().lower() for p in (match or "").split(";") if p.strip()] or DEFAULT_MATCH.split(";")
    files = []
    with os.scandir(path) as entries:
        for e in entries:
            if e.name.startswith(".") or not e.is_file():
                continue
            if any(fnmatch.fnmatchcase(e.name.lower(), p) for p in patterns):
                st = e.stat()
                files.append((e.name, e.path, st.st_mtime_ns, st.st_size))
    if not files:
        raise ValueError(f"{NAME}: no files matching {match!r} in {path}")
    if sort == BY_DATE:
        files.sort(key=lambda f: (f[2], _natural(f[0])))
    else:
        files.sort(key=lambda f: _natural(f[0]))
    return files


def pick_index(pick, seed, count, every=1, salt=""):
    """0-based index for this run. `every` holds each image for that many runs, in every mode."""
    position = SequenceContext(seed).seed // max(1, int(every))
    if pick == SHUFFLED:
        return SequenceContext(position).deck(count, salt=salt)
    if pick == RANDOM:
        return random.Random(f"ab_folder|{position}").randrange(count)  # own RNG, never the global stream
    return SequenceContext(position).step(count)


def fingerprint(folder, match, sort, input_dir):
    """Changes when a matching file is added, removed or touched."""
    try:
        path = resolve_folder(folder, input_dir)
        files = list_images(path, match, sort)
    except (ValueError, OSError):
        return float("nan")  # run, so the error shows
    return hashlib.sha1(repr((path, [(n, m, s) for n, _p, m, s in files])).encode()).hexdigest()


def load_image(path):
    """Like core Load Image: EXIF transpose, RGB, mask = 1 - alpha. First frame only."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    import node_helpers

    img = node_helpers.pillow(Image.open, path)
    img.seek(0)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)
    if img.mode == "I":
        img = img.point(lambda v: v * (1 / 255))
    image = torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0)[None,]
    if "A" in img.getbands():
        mask = 1.0 - torch.from_numpy(np.array(img.getchannel("A")).astype(np.float32) / 255.0)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32)
    return image, mask.unsqueeze(0)


# --- node ---

class ABImageFromFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ABImageFromFolder",
            display_name=NAME,
            category="AB/Loaders",  # NODE_MAINTENANCE.md §6
            description="Loads one image per run from a folder: in order, shuffled without repeats, "
                        "or at random. The seed picks the image, so with the seed on increment each run "
                        "takes the next one. Link the seed to AB Wildcard to pair images with {@…} prompts.",
            search_aliases=["load image folder", "batch images", "image batch from folder", "folder loop",
                            "load images from directory", "dirgir", "image sequence"],
            inputs=[
                io.String.Input("folder", default="",
                                tooltip="Folder inside ComfyUI/input (e.g. shoot_01), or an absolute path "
                                        "on the ComfyUI machine"),
                io.Combo.Input("pick", options=[IN_ORDER, SHUFFLED, RANDOM],
                               tooltip="in order: file 1, 2, 3 …\n"
                                       "shuffled, no repeats: every image once per round, in a shuffled order\n"
                                       "random: any image each run, repeats possible"),
                io.Int.Input("seed", default=0, min=0, max=UINT64,
                             control_after_generate=io.ControlAfterGenerate.increment,
                             tooltip="Picks the image. Same seed, same image. On increment, each run takes "
                                     "the next one"),
                io.Int.Input("every", default=1, min=1, max=10000, advanced=True,
                             tooltip="Next image only after this many runs. Image × 3 prompt variants "
                                     "({@a|b|c}) → 3"),
                io.String.Input("match", default=DEFAULT_MATCH, advanced=True,
                                tooltip="File patterns separated by ; (not case-sensitive)"),
                io.Combo.Input("sort", options=[BY_NAME, BY_DATE], advanced=True,
                               tooltip="name sorts img2 before img10"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="file_name", tooltip="File name without extension"),
                io.String.Output(display_name="file_path"),
                io.Int.Output(display_name="index", tooltip="0-based position of the image in the folder"),
                io.Int.Output(display_name="count", tooltip="Number of matching images"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder, match=DEFAULT_MATCH, sort=BY_NAME, **_kwargs):
        return fingerprint(folder, match, sort, folder_paths.get_input_directory())

    @classmethod
    def execute(cls, folder, pick, seed, every=1, match=DEFAULT_MATCH, sort=BY_NAME):
        path = resolve_folder(folder, folder_paths.get_input_directory())
        files = list_images(path, match, sort)
        index = pick_index(pick, seed, len(files), every, salt=folder.strip())
        name, file_path, _m, _s = files[index]
        image, mask = load_image(file_path)
        return io.NodeOutput(image, mask, os.path.splitext(name)[0], file_path, index, len(files))
