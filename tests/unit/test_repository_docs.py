"""Entry-point documentation must work in a fresh source checkout."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    ROOT / "README.md",
    ROOT / "docs/repository_guide.md",
    ROOT / "docs/current_artifact_inventory.md",
    ROOT / "docs/methanet_positioning_and_claims.md",
    ROOT / "docs/atlas_data_foundation.md",
    ROOT / "docs/knowledge_graph_foundation.md",
    ROOT / "docs/releases/README.md",
    ROOT / "web/emergentbiome-methanet/README.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_entrypoint_local_links_resolve() -> None:
    missing: list[str] = []
    for source in ENTRYPOINTS:
        text = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<"):
                raw_target = raw_target[1:].split(">", 1)[0]
            else:
                raw_target = raw_target.split(maxsplit=1)[0]
            parsed = urlsplit(raw_target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (source.parent / unquote(parsed.path)).resolve()
            line = text.count("\n", 0, match.start()) + 1
            if not target.is_relative_to(ROOT) or not target.exists():
                missing.append(f"{source.relative_to(ROOT)}:{line}: {raw_target}")
    assert not missing, "Broken local documentation links:\n" + "\n".join(missing)
