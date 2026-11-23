"""Tests for folder.py, AB Image from Folder (plan: docs/plan-image-from-folder.md)."""
import importlib.util
import os
import random
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# folder.py imports `.nodes`, so load it as part of the package (any folder name works)
_spec = importlib.util.spec_from_file_location("ab_wildcard_pkg", os.path.join(ROOT, "__init__.py"),
                                               submodule_search_locations=[ROOT])
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["ab_wildcard_pkg"] = _pkg
_spec.loader.exec_module(_pkg)
from ab_wildcard_pkg import folder as f  # noqa: E402


def touch(path, mtime=None, data=b"x"):
    with open(path, "wb") as fh:
        fh.write(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


# --- index math ----------------------------------------------------------------

def test_in_order_steps_and_wraps():
    assert [f.pick_index(f.IN_ORDER, s, 3) for s in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_every_holds_each_image():
    assert [f.pick_index(f.IN_ORDER, s, 3, every=3) for s in range(9)] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [f.pick_index(f.SHUFFLED, s, 4, every=2, salt="x") for s in range(0, 8, 2)] == \
           [f.pick_index(f.SHUFFLED, s, 4, every=2, salt="x") for s in range(1, 8, 2)]
    assert f.pick_index(f.RANDOM, 6, 50, every=3) == f.pick_index(f.RANDOM, 8, 50, every=3)


def test_shuffled_gives_every_index_once_per_round():
    n = 7
    for rnd in range(4):
        got = [f.pick_index(f.SHUFFLED, s, n, salt="shoot") for s in range(rnd * n, (rnd + 1) * n)]
        assert sorted(got) == list(range(n))
    first = [f.pick_index(f.SHUFFLED, s, n, salt="shoot") for s in range(n)]
    assert first != list(range(n))  # actually shuffled (fixed seeds, so stable)


def test_random_reproducible_and_leaves_global_stream_alone():
    a = [f.pick_index(f.RANDOM, s, 100) for s in range(20)]
    assert a == [f.pick_index(f.RANDOM, s, 100) for s in range(20)]
    assert len(set(a)) > 1
    random.seed(1)
    expected = random.random()
    random.seed(1)
    f.pick_index(f.RANDOM, 5, 100)
    assert random.random() == expected


# --- folder resolution ------------------------------------------------------------

def test_relative_resolves_inside_input(tmp_path):
    assert f.resolve_folder(" shoot/a ", str(tmp_path)) == str(tmp_path / "shoot" / "a")


@pytest.mark.parametrize("bad", ["..", "../x", "shoot/../../x"])
def test_dotdot_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="leaves the input folder"):
        f.resolve_folder(bad, str(tmp_path / "input"))


def test_absolute_allowed_and_empty_refused(tmp_path):
    assert f.resolve_folder(str(tmp_path), "/elsewhere") == str(tmp_path)
    with pytest.raises(ValueError, match="no folder set"):
        f.resolve_folder("  ", str(tmp_path))


# --- listing ----------------------------------------------------------------------

def names(files):
    return [n for n, *_ in files]


def test_natural_sort(tmp_path):
    for n in ["img10.png", "img2.png", "Img1.png", "a.png"]:
        touch(tmp_path / n)
    assert names(f.list_images(str(tmp_path))) == ["a.png", "Img1.png", "img2.png", "img10.png"]


def test_sort_by_date(tmp_path):
    touch(tmp_path / "b.png", mtime=1000)
    touch(tmp_path / "a.png", mtime=3000)
    touch(tmp_path / "c.png", mtime=2000)
    assert names(f.list_images(str(tmp_path), sort=f.BY_DATE)) == ["b.png", "c.png", "a.png"]


def test_match_filter(tmp_path):
    for n in ["a.PNG", "b.jpg", "c.txt", ".hidden.png", "d.webp"]:
        touch(tmp_path / n)
    (tmp_path / "sub.png").mkdir()  # folders never count, nor their contents
    touch(tmp_path / "sub.png" / "e.png")
    assert names(f.list_images(str(tmp_path))) == ["a.PNG", "b.jpg", "d.webp"]
    assert names(f.list_images(str(tmp_path), match=" *.jpg ; *.TXT")) == ["b.jpg", "c.txt"]


def test_empty_and_missing_folder_errors(tmp_path):
    touch(tmp_path / "notes.txt")
    with pytest.raises(ValueError) as e:
        f.list_images(str(tmp_path), match="*.png")
    assert str(tmp_path) in str(e.value) and "*.png" in str(e.value)
    with pytest.raises(ValueError, match="folder not found"):
        f.list_images(str(tmp_path / "nope"))


# --- cache fingerprint ------------------------------------------------------------

def test_fingerprint_changes_on_add_and_touch(tmp_path):
    touch(tmp_path / "a.png", mtime=1000)
    fp = lambda: f.fingerprint(str(tmp_path), f.DEFAULT_MATCH, f.BY_NAME, "/unused")  # noqa: E731
    first = fp()
    assert fp() == first
    touch(tmp_path / "notes.txt")  # not matched: no re-run
    assert fp() == first
    touch(tmp_path / "b.png", mtime=1000)
    added = fp()
    assert added != first
    os.utime(tmp_path / "a.png", (2000, 2000))
    assert fp() != added


def test_fingerprint_on_error_forces_run(tmp_path):
    assert f.fingerprint(str(tmp_path / "nope"), f.DEFAULT_MATCH, f.BY_NAME, "/x") != \
           f.fingerprint(str(tmp_path / "nope"), f.DEFAULT_MATCH, f.BY_NAME, "/x")  # nan


# --- pick: selected and the thumbnail grid ------------------------------------------

def test_selected_finds_file_by_name(tmp_path):
    for n in ["img1.png", "img2.png", "img10.png"]:
        touch(tmp_path / n)
    files = f.list_images(str(tmp_path))
    assert f.index_of(files, "img10.png") == 2
    touch(tmp_path / "img0.png")  # a new file moves indexes, not the choice
    assert f.list_images(str(tmp_path))[f.index_of(f.list_images(str(tmp_path)), "img10.png")][0] == "img10.png"
    with pytest.raises(ValueError, match="not in the folder"):
        f.index_of(files, "gone.png")
    with pytest.raises(ValueError, match="no file is set"):
        f.index_of(files, "")


def test_image_file_serves_only_listed_files(tmp_path):
    (tmp_path / "shoot").mkdir()
    touch(tmp_path / "shoot" / "a.png")
    touch(tmp_path / "shoot" / ".hidden.png")
    touch(tmp_path / "shoot" / "notes.txt")
    touch(tmp_path / "outside.png")
    folder = str(tmp_path / "shoot")
    assert f.image_file(folder, "a.png") == os.path.join(folder, "a.png")
    for bad in ["../outside.png", "sub/a.png", ".hidden.png", "notes.txt", "missing.png", "", ".."]:
        with pytest.raises(ValueError):
            f.image_file(folder, bad)
    assert f.image_file(folder, "notes.txt", match="*.txt")


def test_thumbnail_is_small_jpeg(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGBA", (800, 400), (255, 0, 0, 128)).save(tmp_path / "a.png")
    data = f.thumbnail(str(tmp_path / "a.png"), 1)
    assert data[:2] == b"\xff\xd8"  # JPEG
    from io import BytesIO
    assert Image.open(BytesIO(data)).size == (f.THUMB, f.THUMB // 2)


# --- loading ----------------------------------------------------------------------

def test_load_image_mask_from_alpha(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "node_helpers",
                        types.SimpleNamespace(pillow=lambda fn, arg: fn(arg)))
    Image.new("RGBA", (4, 2), (255, 0, 0, 0)).save(tmp_path / "a.png")
    Image.new("RGB", (4, 2), (0, 255, 0)).save(tmp_path / "b.jpg")

    image, mask = f.load_image(str(tmp_path / "a.png"))
    assert tuple(image.shape) == (1, 2, 4, 3) and tuple(mask.shape) == (1, 2, 4)
    assert float(image[0, 0, 0, 0]) == 1.0 and float(mask.min()) == 1.0  # alpha 0 -> mask 1

    image, mask = f.load_image(str(tmp_path / "b.jpg"))
    assert tuple(image.shape) == (1, 2, 4, 3) and tuple(mask.shape) == (1, 64, 64)
    assert float(mask.max()) == 0.0
