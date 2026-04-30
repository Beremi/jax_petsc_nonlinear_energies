#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import FIGURES_ROOT, PAPER_ROOT, TABLES_ROOT, ensure_paper_dirs


INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
INPUT_IF_EXISTS_RE = re.compile(r"\\InputIfFileExists\s*\{([^{}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])*\s*\{([^{}]+)\}")


def _strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        escaped = False
        kept: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            kept.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(kept))
    return "\n".join(lines)


def _tex_path(raw: str) -> str:
    path = raw.strip()
    if not Path(path).suffix:
        path += ".tex"
    return path


def _resolve_tex_path(raw: str, current_dir: Path) -> Path:
    path = Path(_tex_path(raw))
    candidates = [PAPER_ROOT / path]
    if not path.is_absolute():
        candidates.append(current_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _figure_name(raw: str) -> str:
    path = Path(raw.strip())
    if not path.suffix:
        path = path.with_suffix(".pdf")
    return path.name


def _table_name(raw: str) -> str | None:
    path = Path(_tex_path(raw))
    parts = path.parts
    if len(parts) >= 3 and parts[-3:-1] == ("tables", "generated"):
        return path.name
    return None


def _collect_tex_assets(path: Path, *, seen: set[Path] | None = None) -> tuple[set[str], set[str], set[Path]]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return set(), set(), seen
    seen.add(path)
    if not path.exists():
        return set(), set(), seen

    text = _strip_tex_comments(path.read_text(encoding="utf-8"))
    figures = {_figure_name(match.group(1)) for match in GRAPHICS_RE.finditer(text)}
    tables: set[str] = set()
    input_names = [match.group(1) for match in INCLUDE_RE.finditer(text)]
    input_names.extend(match.group(1) for match in INPUT_IF_EXISTS_RE.finditer(text))
    for raw in input_names:
        table_name = _table_name(raw)
        if table_name is not None:
            tables.add(table_name)
            continue
        child_figures, child_tables, seen = _collect_tex_assets(_resolve_tex_path(raw, path.parent), seen=seen)
        figures.update(child_figures)
        tables.update(child_tables)
    return figures, tables, seen


def _manifest_assets(figures_dir: Path) -> set[str]:
    manifest_path = figures_dir / "manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets: set[str] = set()
    for key in ("generated_assets", "copied_assets"):
        for name in manifest.get(key, []):
            if isinstance(name, str):
                assets.add(Path(name).name)
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the paper asset generation produced the expected files.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_ROOT)
    parser.add_argument("--tex", type=Path, default=PAPER_ROOT / "main.tex")
    args = parser.parse_args()
    ensure_paper_dirs()
    required_figures, required_tables, _seen = _collect_tex_assets(args.tex)
    missing: list[str] = []
    for name in sorted(required_figures):
        path = args.figures_dir / name
        if not path.exists():
            missing.append(str(path))
    for name in sorted(required_tables):
        path = args.tables_dir / name
        if not path.exists():
            missing.append(str(path))
    manifest_assets = _manifest_assets(args.figures_dir)
    untracked_figures = sorted(required_figures - manifest_assets)
    if missing:
        raise SystemExit("Missing paper assets:\n" + "\n".join(missing))
    if untracked_figures:
        raise SystemExit("TeX-included figures missing from figure manifest:\n" + "\n".join(untracked_figures))
    print(f"Paper assets validated ({len(required_figures)} figures, {len(required_tables)} tables).")


if __name__ == "__main__":
    main()
