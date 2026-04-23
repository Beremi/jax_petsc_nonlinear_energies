#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import requests

from common import FULLTEXT_ROOT, LITERATURE_ROOT, PAPER_ROOT, ensure_paper_dirs, read_json, write_text


EXPECTED_UNCITED_KEYS = {"conn2000trust"}
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)
CITE_RE = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^]]*\])*\{([^}]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def extract_cited_keys(paper_root: Path) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    main_tex = paper_root / "main.tex"
    tex_paths = [main_tex]
    main_text = main_tex.read_text(encoding="utf-8")
    for match in INPUT_RE.finditer(main_text):
        rel_path = match.group(1).strip()
        tex_paths.append((paper_root / rel_path).with_suffix(".tex"))
    for path in tex_paths:
        text = path.read_text(encoding="utf-8")
        for match in CITE_RE.finditer(text):
            for raw_key in match.group(1).split(","):
                key = raw_key.strip()
                if key and key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def parse_bibtex(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, Any]] = {}
    index = 0
    length = len(text)
    while index < length:
        at = text.find("@", index)
        if at == -1:
            break
        brace = text.find("{", at)
        if brace == -1:
            raise ValueError(f"Malformed BibTeX entry near byte offset {at}")
        entry_type = text[at + 1 : brace].strip().lower()
        index = brace + 1
        key_end = text.find(",", index)
        if key_end == -1:
            raise ValueError(f"Missing BibTeX key separator near byte offset {index}")
        key = text[index:key_end].strip()
        index = key_end + 1
        fields: dict[str, str] = {}
        while index < length:
            while index < length and text[index].isspace():
                index += 1
            if index >= length:
                raise ValueError(f"Unterminated BibTeX entry {key!r}")
            if text[index] == "}":
                index += 1
                break
            field_end = text.find("=", index)
            if field_end == -1:
                raise ValueError(f"Missing field delimiter in BibTeX entry {key!r}")
            field_name = text[index:field_end].strip().lower()
            index = field_end + 1
            while index < length and text[index].isspace():
                index += 1
            if index >= length:
                raise ValueError(f"Missing value for BibTeX field {field_name!r} in {key!r}")
            value, index = read_bib_value(text, index)
            fields[field_name] = value.strip()
            while index < length and text[index].isspace():
                index += 1
            if index < length and text[index] == ",":
                index += 1
        entries[key] = {"entry_type": entry_type, "fields": fields}
    return entries


def read_bib_value(text: str, index: int) -> tuple[str, int]:
    if text[index] == "{":
        depth = 0
        start = index + 1
        index += 1
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    return text[start:index], index + 1
                depth -= 1
            index += 1
        raise ValueError("Unterminated braced BibTeX value")
    if text[index] == '"':
        start = index + 1
        index += 1
        escaped = False
        while index < len(text):
            char = text[index]
            if char == '"' and not escaped:
                return text[start:index], index + 1
            escaped = (char == "\\") and not escaped
            if char != "\\":
                escaped = False
            index += 1
        raise ValueError("Unterminated quoted BibTeX value")
    start = index
    while index < len(text) and text[index] not in ",}":
        index += 1
    return text[start:index], index


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped.lower()


def parse_isbns(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {match.group(0) for match in re.finditer(r"[0-9Xx-]{10,17}", value)}


def bib_title_to_text(value: str) -> str:
    text = value
    replacements = {
        r"{\o}": "ø",
        r"{\O}": "Ø",
        r"{\ae}": "æ",
        r"{\AE}": "Æ",
        r"{\aa}": "å",
        r"{\AA}": "Å",
        r"{\c{s}}": "ş",
        r"{\c{S}}": "Ş",
        r"\&": "&",
        "~": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def markdown_link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def markdown_cell(text: str) -> str:
    return text.replace("|", r"\|")


def download_fulltext(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}
    if "asep.lib.cas.cz" in url:
        headers["Referer"] = "https://asep.lib.cas.cz/"
    response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"Downloaded empty payload from {url}")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(destination)


def render_table(rows: list[str], header: str, columns: str) -> list[str]:
    return [header, "", columns, *rows, ""]


def collect_supplemental_keys(
    manifest_entries: dict[str, dict[str, Any]], cited_key_set: set[str]
) -> list[str]:
    supplemental_keys: list[str] = []
    invalid_uncited_keys: list[str] = []
    for key, entry in manifest_entries.items():
        is_supplemental = bool(entry.get("supplemental"))
        if key in cited_key_set:
            if is_supplemental:
                raise SystemExit(f"Cited key {key!r} must not be marked supplemental in the manifest")
            continue
        if is_supplemental:
            supplemental_keys.append(key)
        else:
            invalid_uncited_keys.append(key)
    if invalid_uncited_keys:
        raise SystemExit(
            "Manifest contains uncited entries that are not marked supplemental: "
            f"{sorted(invalid_uncited_keys)}"
        )
    return supplemental_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper literature source index and download public full texts.")
    parser.add_argument("--bib", type=Path, default=PAPER_ROOT / "references.bib")
    parser.add_argument("--manifest", type=Path, default=LITERATURE_ROOT / "manifest.json")
    parser.add_argument("--out-md", type=Path, default=LITERATURE_ROOT / "sources.md")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    ensure_paper_dirs()
    FULLTEXT_ROOT.mkdir(parents=True, exist_ok=True)

    cited_keys = extract_cited_keys(PAPER_ROOT)
    if not cited_keys:
        raise SystemExit("No cited keys were extracted from the paper sources")
    for key in EXPECTED_UNCITED_KEYS:
        if key in cited_keys:
            raise SystemExit(f"Key {key!r} must remain uncited but was extracted from the paper")

    bib_entries = parse_bibtex(args.bib)
    manifest_entries = read_json(args.manifest)["entries"]

    cited_key_set = set(cited_keys)
    manifest_key_set = set(manifest_entries)
    missing = sorted(cited_key_set - manifest_key_set)
    if missing:
        raise SystemExit(f"Manifest is missing cited keys: {missing}")
    supplemental_keys = collect_supplemental_keys(manifest_entries, cited_key_set)
    keys_to_render = cited_keys + supplemental_keys

    available_rows: list[str] = []
    private_rows: list[str] = []
    unavailable_rows: list[str] = []
    covered_keys: list[str] = []

    for key in keys_to_render:
        if key not in bib_entries:
            raise SystemExit(f"Literature key {key!r} is missing from {args.bib}")
        fields = bib_entries[key]["fields"]
        manifest = manifest_entries[key]

        manifest_doi = normalize_doi(manifest.get("doi"))
        bib_doi = normalize_doi(fields.get("doi"))
        if manifest_doi != bib_doi:
            raise SystemExit(f"DOI mismatch for {key!r}: manifest={manifest_doi!r}, bib={bib_doi!r}")

        expected_isbns = set(manifest.get("isbns", []))
        if expected_isbns:
            actual_isbns = parse_isbns(fields.get("isbn"))
            missing_isbns = sorted(expected_isbns - actual_isbns)
            if missing_isbns:
                raise SystemExit(f"ISBN mismatch for {key!r}: missing {missing_isbns}, actual={sorted(actual_isbns)}")

        canonical_source_url = manifest["canonical_source_url"]
        bib_url = fields.get("url")
        if bib_url != canonical_source_url:
            raise SystemExit(f"URL mismatch for {key!r}: manifest={canonical_source_url!r}, bib={bib_url!r}")

        title = markdown_cell(bib_title_to_text(fields["title"]))
        doi_cell = ""
        if manifest_doi:
            doi_cell = markdown_link(fields["doi"], f"https://doi.org/{fields['doi']}")
        source_cell = markdown_link(manifest["canonical_source_label"], canonical_source_url)

        fulltext_url = manifest.get("fulltext_url")
        local_filename = manifest.get("local_filename")
        note = markdown_cell(manifest["notes"])
        if fulltext_url:
            if not local_filename:
                raise SystemExit(f"Manifest entry {key!r} is missing local_filename for downloadable full text")
            destination = FULLTEXT_ROOT / local_filename
            if not args.skip_download:
                download_fulltext(fulltext_url, destination)
            if not destination.exists():
                raise SystemExit(f"Expected downloaded full text for {key!r} at {destination}")
            local_link = markdown_link(local_filename, f"fulltext/{local_filename}")
            fulltext_cell = markdown_link(manifest["fulltext_label"], fulltext_url)
            available_rows.append(
                f"| `{key}` | {title} | {doi_cell} | {source_cell} | {fulltext_cell} | {local_link} |"
            )
        elif local_filename:
            destination = FULLTEXT_ROOT / local_filename
            if not destination.exists():
                raise SystemExit(f"Expected local non-public full text for {key!r} at {destination}")
            local_link = markdown_link(local_filename, f"fulltext/{local_filename}")
            private_rows.append(
                f"| `{key}` | {title} | {doi_cell} | {source_cell} | {local_link} | {note} |"
            )
        else:
            unavailable_rows.append(f"| `{key}` | {title} | {doi_cell} | {source_cell} | {note} |")
        covered_keys.append(key)

    if sorted(covered_keys) != sorted(keys_to_render):
        raise SystemExit("Coverage mismatch between expected literature keys and generated literature tables")
    if len(set(covered_keys)) != len(keys_to_render):
        raise SystemExit("Some literature keys appeared more than once across the generated literature tables")

    lines = [
        "# Literature Sources",
        "",
        (
            "Generated by `python scripts/generate_literature_sources.py` from "
            "`paper/references.bib` and `paper/literature/manifest.json`, covering cited sources "
            "plus any manifest entries marked as supplemental literature."
        ),
        "",
        *render_table(
            available_rows,
            "## Available full text",
            "| Key | Title | DOI | Canonical source | Public full-text link | Local copy |\n| --- | --- | --- | --- | --- | --- |",
        ),
        *render_table(
            private_rows,
            "## Non-public full text available",
            "| Key | Title | DOI | Canonical source | Local copy | Notes |\n| --- | --- | --- | --- | --- | --- |",
        ),
        *render_table(
            unavailable_rows,
            "## Full text not available",
            "| Key | Title | DOI | Canonical source | Notes |\n| --- | --- | --- | --- | --- |",
        ),
    ]
    write_text(args.out_md, "\n".join(lines).rstrip() + "\n")
    print(
        f"Generated {args.out_md} with {len(available_rows)} public entries, "
        f"{len(private_rows)} non-public local entries, and {len(unavailable_rows)} unavailable entries."
    )


if __name__ == "__main__":
    main()
