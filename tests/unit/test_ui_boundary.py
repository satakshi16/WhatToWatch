"""P1-13: the Streamlit UI talks to the backend over HTTP only.

It must never import `app.db` (acceptance criterion); we go further and forbid any
`app.*` import so the UI stays a pure API client.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
UI_FILES = sorted(UI_DIR.rglob("*.py"))


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_ui_package_has_the_expected_files() -> None:
    names = {p.name for p in UI_FILES}
    assert {"streamlit_app.py", "api_client.py", "attribution.py"} <= names


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_ui_never_imports_app_db(path: Path) -> None:
    bad = {m for m in imported_modules(path) if m == "app.db" or m.startswith("app.db.")}
    assert not bad, f"{path.name} imports {sorted(bad)}"


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_ui_never_imports_the_backend(path: Path) -> None:
    bad = {m for m in imported_modules(path) if m == "app" or m.startswith("app.")}
    assert not bad, f"{path.name} imports {sorted(bad)}"


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_ui_has_no_dynamic_imports(path: Path) -> None:
    text = path.read_text("utf-8")
    assert "importlib.import_module" not in text
    assert "__import__(" not in text


def test_checker_catches_app_db_imports(tmp_path: Path) -> None:
    sample = tmp_path / "x.py"
    sample.write_text("import os\nfrom app.db.models import User\nfrom app import db\n")
    assert {"app.db.models", "app.db"} <= imported_modules(sample)
