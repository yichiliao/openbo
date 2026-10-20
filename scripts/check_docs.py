"""Check local documentation links and the first four teaching notebooks."""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0[1-4]_*.ipynb"))


def anchors(path: Path) -> set[str]:
    """GitHub-style heading IDs, including suffixes for repeated headings."""
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    found = set(re.findall(r'(?:id|name)=["\']([^"\']+)', text))
    seen: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.M):
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", html.unescape(heading).strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        found.add(f"{slug}-{count}" if count else slug)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-notebooks", action="store_true", help="Execute each lesson in a fresh process.")
    args = parser.parse_args()
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").rglob("*.md"))]
    errors = []
    for path in documents:
        source = path.read_text(encoding="utf-8")
        # Ignore code fences, which can contain illustrative Markdown syntax.
        source = re.sub(r"```.*?```", "", source, flags=re.S)
        for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source):
            target = urlsplit(raw.strip().split()[0].strip("<>"))
            if target.scheme or target.netloc:
                continue
            destination = path.parent / unquote(target.path) if target.path else path
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {raw}")
            elif target.fragment and destination.suffix == ".md" and unquote(target.fragment) not in anchors(destination):
                errors.append(f"{path.relative_to(ROOT)}: missing heading in {raw}")
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not notebook.get("cells"):
            errors.append(f"{path.name}: expected a nonempty v4 notebook")
            continue
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        code = "\n\n".join(cells)
        ast.parse(code, filename=path.name)
        if args.execute_notebooks:
            print(f"Executing {path.name}", flush=True)
            subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=180)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked links in {len(documents)} Markdown files and {len(NOTEBOOKS)} teaching notebooks.")


if __name__ == "__main__":
    main()
