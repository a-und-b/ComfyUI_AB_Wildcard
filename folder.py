"""AB Image from Folder: one image per run from a folder, picked by the seed.

Nothing is remembered between runs: the index comes from the seed alone, with the
same seed logic as {~ ! @} in AB Wildcard (SequenceContext). Same seed, same image,
so ComfyUI's cache, a restart and API runs all behave. The one exception is
pick "selected": the file named in the `file` widget, set by a click on the
node's thumbnail grid (js/ab_image_folder.js, served by the routes below).
Plan: docs/plan-image-from-folder.md
"""
import asyncio
import fnmatch
import functools
import hashlib
import os
import random
import re
from io import BytesIO

import folder_paths
import server
from aiohttp import web
from comfy_api.latest import io

from .nodes import SequenceContext

IN_ORDER, SHUFFLED, RANDOM, SELECTED = "in order", "shuffled, no repeats", "random", "selected"
BY_NAME, BY_DATE = "name", "date modified"
DEFAULT_MATCH = "*.png;*.jpg;*.jpeg;*.webp"
UINT64 = 0xffffffffffffffff
NAME = "AB Image from Folder"
THUMB = 160


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


def _patterns(match):
    return [p.strip().lower() for p in (match or "").split(";") if p.strip()] or DEFAULT_MATCH.split(";")


def _wanted(name, patterns):
    return not name.startswith(".") and any(fnmatch.fnmatchcase(name.lower(), p) for p in patterns)


def list_images(path, match=DEFAULT_MATCH, sort=BY_NAME):
    """Matching files in path (not recursive, no hidden files) as (name, path, mtime_ns, size)."""
    if not os.path.isdir(path):
        raise ValueError(f"{NAME}: folder not found: {path}")
    patterns = _patterns(match)
    files = []
    with os.scandir(path) as entries:
        for e in entries:
            if e.is_file() and _wanted(e.name, patterns):
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


def index_of(files, name, match=DEFAULT_MATCH):
    """Index of the file picked by name (pick: selected)."""
    for i, f in enumerate(files):
        if f[0] == name:
            return i
    if not name:
        raise ValueError(f"{NAME}: pick is 'selected' but no file is set. Click a thumbnail on the node.")
    raise ValueError(f"{NAME}: {name!r} is not in the folder (match {match!r}). Click a thumbnail to pick another.")


def image_file(path, name, match=DEFAULT_MATCH):
    """Full path of one file the listing of path would contain, without listing the folder."""
    full = os.path.join(path, name or "")
    if os.path.basename(name or "") != name or not _wanted(name, _patterns(match)) or not os.path.isfile(full):
        raise ValueError(f"{NAME}: no file {name!r} matching {match!r} in {path}")
    return full


@functools.lru_cache(maxsize=1024)  # ponytail: ~10 KB a thumb, so ≤ ~10 MB; a disk cache if folders get huge
def thumbnail(path, mtime_ns, size=THUMB):
    """JPEG bytes, longest side `size`. mtime_ns is only the cache key: a changed file gets a new thumb."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img.draft("RGB", (size * 2, size * 2))  # fast JPEG decode at reduced size; no-op for other formats
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size))
        buf = BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=80)
    return buf.getvalue()


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
                io.Combo.Input("pick", options=[IN_ORDER, SHUFFLED, RANDOM, SELECTED],
                               tooltip="in order: file 1, 2, 3 …\n"
                                       "shuffled, no repeats: every image once per round, in a shuffled order\n"
                                       "random: any image each run, repeats possible\n"
                                       "selected: the file below, set by a click on a thumbnail"),
                io.String.Input("file", default="",
                                tooltip="File name for pick: selected. A click on a thumbnail sets it"),
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
    def execute(cls, folder, pick, seed, file="", every=1, match=DEFAULT_MATCH, sort=BY_NAME):
        path = resolve_folder(folder, folder_paths.get_input_directory())
        files = list_images(path, match, sort)
        if pick == SELECTED:
            index = index_of(files, file, match)
        else:
            index = pick_index(pick, seed, len(files), every, salt=folder.strip())
        name, file_path, _m, _s = files[index]
        image, mask = load_image(file_path)
        return io.NodeOutput(image, mask, os.path.splitext(name)[0], file_path, index, len(files),
                             ui={"ab_file": [name]})  # the grid marks the file this run used


# --- routes for the thumbnail grid (js/ab_image_folder.js) ---

routes = server.PromptServer.instance.routes


def _folder(query):
    return resolve_folder(query.get("folder", ""), folder_paths.get_input_directory())


@routes.get("/ab_wildcard/folder/list")
async def folder_list(request):
    q = request.query
    try:
        files = list_images(_folder(q), q.get("match") or DEFAULT_MATCH, q.get("sort", BY_NAME))
    except (ValueError, OSError) as e:
        return web.json_response({"error": str(e)})
    return web.json_response({"files": [[n, str(m)] for n, _p, m, _s in files]})  # str: ns exceed JS numbers


@routes.get("/ab_wildcard/folder/thumb")
async def folder_thumb(request):
    q = request.query
    try:
        path = image_file(_folder(q), q.get("name", ""), q.get("match") or DEFAULT_MATCH)
        data = await asyncio.to_thread(thumbnail, path, os.stat(path).st_mtime_ns)
    except (ValueError, OSError) as e:
        return web.Response(status=404, text=str(e))
    # the URL carries the mtime, so the browser may keep it
    return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "max-age=604800"})
