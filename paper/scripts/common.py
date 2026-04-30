from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
BUILD_ROOT = PAPER_ROOT / "build"
FIGURES_ROOT = PAPER_ROOT / "figures" / "generated"
TABLES_ROOT = PAPER_ROOT / "tables" / "generated"
SCRIPTS_ROOT = PAPER_ROOT / "scripts"
LITERATURE_ROOT = PAPER_ROOT / "literature"
FULLTEXT_ROOT = LITERATURE_ROOT / "fulltext"
LAYOUT_JSON = BUILD_ROOT / "layout.json"


def ensure_paper_dirs() -> None:
    for path in (BUILD_ROOT, FIGURES_ROOT, TABLES_ROOT, SCRIPTS_ROOT, LITERATURE_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_layout() -> dict[str, float]:
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"Layout JSON missing: {LAYOUT_JSON}")
    payload = read_json(LAYOUT_JSON)
    return {
        "columnwidth_pt": float(payload["columnwidth_pt"]),
        "textwidth_pt": float(payload["textwidth_pt"]),
        "columnwidth_in": float(payload["columnwidth_in"]),
        "textwidth_in": float(payload["textwidth_in"]),
    }


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def pt_to_in(value_pt: float) -> float:
    return float(value_pt) / 72.27


def column_figure_size(
    layout: dict[str, float],
    *,
    width_scale: float = 1.0,
    height_ratio: float | None = None,
    height_in: float | None = None,
) -> tuple[float, float]:
    width = layout["columnwidth_in"] * width_scale
    if height_in is not None:
        return width, height_in
    if height_ratio is None:
        height_ratio = 0.62
    return width, width * height_ratio


def text_figure_size(
    layout: dict[str, float],
    *,
    width_scale: float = 1.0,
    height_ratio: float | None = None,
    height_in: float | None = None,
) -> tuple[float, float]:
    width = layout["textwidth_in"] * width_scale
    if height_in is not None:
        return width, height_in
    if height_ratio is None:
        height_ratio = 0.38
    return width, width * height_ratio


def paper_width_in(layout: dict[str, float], preset: str = "full") -> float:
    preset = str(preset)
    scales = {
        "subfigure": 0.46,
        "full": 1.0,
        "medium": 0.84,
        "narrow": 0.72,
    }
    if preset not in scales:
        raise ValueError(f"Unsupported figure preset {preset!r}")
    return layout["textwidth_in"] * scales[preset]


def paper_figure_size(
    layout: dict[str, float],
    *,
    preset: str = "full",
    height_ratio: float | None = None,
    height_in: float | None = None,
) -> tuple[float, float]:
    width = paper_width_in(layout, preset)
    if height_in is not None:
        return width, height_in
    if height_ratio is None:
        height_ratio = 0.40
    return width, width * height_ratio


def configure_paper_matplotlib(font_size: float = 10.0):
    from experiments.analysis.docs_assets.common import configure_matplotlib

    plt = configure_matplotlib()
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "axes.titlepad": 2.5,
        }
    )
    return plt


def save_pdf_and_png(fig, pdf_path: Path, *, png_dpi: int = 240) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", dpi=600)
    fig.savefig(pdf_path.with_suffix(".png"), format="png", dpi=png_dpi)


def copy_asset(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
