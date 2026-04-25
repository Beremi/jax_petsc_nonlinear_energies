from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(ref))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(cand))
    return float(np.linalg.norm(ref - cand) / denom)


def parse_markdown_pipe_table(text: str, heading: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        if raw.strip() != str(heading).strip():
            continue
        header_idx = idx + 2
        sep_idx = idx + 3
        if header_idx >= len(lines) or sep_idx >= len(lines):
            break
        header_line = lines[header_idx].strip()
        sep_line = lines[sep_idx].strip()
        if "|" not in header_line or "|" not in sep_line:
            break
        headers = [cell.strip() for cell in header_line.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for row_raw in lines[sep_idx + 1 :]:
            row = row_raw.strip()
            if not row or "|" not in row:
                break
            values = [cell.strip() for cell in row.strip("|").split("|")]
            if len(values) != len(headers):
                break
            rows.append(dict(zip(headers, values, strict=True)))
        return rows
    raise ValueError(f"Could not find markdown table under heading {heading!r}")


def critical_lambda_schedule_proxy(rows: list[dict[str, object]], *, lambda_key: str = "lambda_value", success_key: str = "solver_success") -> float:
    completed = [
        float(row[lambda_key])
        for row in rows
        if bool(row.get(success_key, False)) and math.isfinite(float(row.get(lambda_key, float("nan"))))
    ]
    if not completed:
        return float("nan")
    return float(max(completed))


def curve_relative_l2(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
) -> float:
    ref_x = np.asarray(reference_x, dtype=np.float64).ravel()
    ref_y = np.asarray(reference_y, dtype=np.float64).ravel()
    cand_x = np.asarray(candidate_x, dtype=np.float64).ravel()
    cand_y = np.asarray(candidate_y, dtype=np.float64).ravel()
    if ref_x.size == 0 or cand_x.size == 0:
        return float("nan")
    if ref_x.shape != cand_x.shape or not np.allclose(ref_x, cand_x, atol=1.0e-12, rtol=0.0):
        cand_interp = np.interp(ref_x, cand_x, cand_y)
    else:
        cand_interp = cand_y
    return relative_l2(ref_y, cand_interp)


def compute_boundary_profile(
    coords_ref: np.ndarray,
    coords_final: np.ndarray,
    *,
    y_band_fraction: float = 0.04,
    z_quantile: float = 0.6,
) -> dict[str, np.ndarray]:
    ref = np.asarray(coords_ref, dtype=np.float64)
    final = np.asarray(coords_final, dtype=np.float64)
    disp = final - ref
    y_mid = float(0.5 * (np.min(ref[:, 1]) + np.max(ref[:, 1])))
    y_half = float(max(np.max(ref[:, 1]) - np.min(ref[:, 1]), 1.0e-12) * float(y_band_fraction))
    z_cut = float(np.quantile(ref[:, 2], float(z_quantile)))
    mask = (np.abs(ref[:, 1] - y_mid) <= y_half) & (ref[:, 2] >= z_cut)
    selected = np.where(mask)[0]
    if selected.size < 3:
        selected = np.argsort(ref[:, 2])[-max(16, min(128, ref.shape[0] // 16)) :]
    order = selected[np.argsort(ref[selected, 0])]
    u_mag = np.linalg.norm(disp[order], axis=1)
    return {
        "x": np.asarray(ref[order, 0], dtype=np.float64),
        "z": np.asarray(ref[order, 2], dtype=np.float64),
        "u_mag": np.asarray(u_mag, dtype=np.float64),
        "indices": np.asarray(order, dtype=np.int64),
    }


def acceptance_flags(
    *,
    critical_lambda_rel_diff: float,
    umax_curve_rel_l2: float,
    endpoint_disp_rel_l2: float,
) -> dict[str, bool]:
    return {
        "critical_lambda_pass": bool(
            math.isfinite(float(critical_lambda_rel_diff))
            and float(critical_lambda_rel_diff) <= 0.03
        ),
        "umax_curve_pass": bool(
            math.isfinite(float(umax_curve_rel_l2))
            and float(umax_curve_rel_l2) <= 0.05
        ),
        "endpoint_disp_pass": bool(
            math.isfinite(float(endpoint_disp_rel_l2))
            and float(endpoint_disp_rel_l2) <= 0.10
        ),
    }


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
