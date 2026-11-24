from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def _norm(spec: str) -> str:
    # Strip inline comments, whitespace, and normalize package name casing
    spec = spec.split("#", 1)[0].strip()
    return re.sub(r"\s+", "", spec).lower()


def test_dependencies_match_requirements():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    py_deps = {_norm(d) for d in pyproject.get("project", {}).get("dependencies", []) if _norm(d)}

    req_file = ROOT / "requirements.txt"
    req_deps = set()
    if req_file.exists():
        for line in req_file.read_text(encoding="utf-8").splitlines():
            line = _norm(line)
            if line and not line.startswith("-"):
                req_deps.add(line)

    assert py_deps == req_deps, (
        f"pyproject.toml and requirements.txt are out of sync:\n"
        f"  only in pyproject.toml:   {sorted(py_deps - req_deps)}\n"
        f"  only in requirements.txt: {sorted(req_deps - py_deps)}"
    )
