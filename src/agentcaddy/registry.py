"""Scan every skill, agent, MCP server and context file into one registry.

Capabilities are data: this file is the only thing that has to change when a new skill is added
(it does not, because it scans folders). Run: python -m agentcaddy.registry
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HOME = Path.home()
BODY_HEAD_CHARS = 600
CHARS_PER_TOKEN = 4

SKILL_ROOTS: dict[str, Path] = {
    "claude": HOME / ".claude" / "skills",
    "agents": HOME / ".agents" / "skills",
    "codex": HOME / ".codex" / "skills",
    "hermes": HOME / "AppData" / "Local" / "hermes" / "skills",
}
ARCHIVE_ROOT = HOME / ".claude" / "skills-archive"
PLUGIN_CACHE = HOME / ".claude" / "plugins" / "cache"
AGENTS_DIR = HOME / ".claude" / "agents"
CLAUDE_JSON = HOME / ".claude.json"
CONTEXT_FILES = [
    HOME / ".agents" / "AGENTS.md",
    HOME / ".claude" / "CLAUDE.md",
]
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "registry.json"


@dataclass
class Capability:
    id: str
    kind: str  # skill | agent | mcp_server | context
    name: str
    description: str
    body_head: str
    locations: list[str] = field(default_factory=list)
    status: str = "active"  # active | archived
    tokens_description: int = 0
    tokens_body: int = 0


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML frontmatter reader: handles `key: value`, quoted values and `>`/`|` blocks."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    key: str | None = None
    for line in match.group(1).splitlines():
        top = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if top and not line.startswith(" "):
            key, value = top.group(1), top.group(2).strip()
            meta[key] = "" if value in (">", "|", ">-", "|-") else value.strip("\"'")
        elif key and line.strip():
            meta[key] = (meta[key] + " " + line.strip()).strip()
    return meta, match.group(2)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def content_hash(name: str, description: str, body: str) -> str:
    return hashlib.sha1(f"{name}\n{description}\n{body}".encode("utf-8", "ignore")).hexdigest()[:12]


def read_skill(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta, body = parse_frontmatter(text)
    name = meta.get("name") or path.parent.name
    return name, meta.get("description", ""), body.strip()


def add(registry: dict[str, Capability], kind: str, name: str, description: str, body: str,
        location: str, status: str) -> None:
    cap_id = f"{kind}:{content_hash(name, description, body)}"
    existing = registry.get(cap_id)
    if existing:
        existing.locations.append(location)
        if status == "active":
            existing.status = "active"
        return
    registry[cap_id] = Capability(
        id=cap_id, kind=kind, name=name, description=description,
        body_head=body[:BODY_HEAD_CHARS], locations=[location], status=status,
        tokens_description=estimate_tokens(description), tokens_body=estimate_tokens(body),
    )


def scan_skills(registry: dict[str, Capability]) -> None:
    for tool, root in SKILL_ROOTS.items():
        for path in root.glob("*/SKILL.md") if root.exists() else []:
            name, description, body = read_skill(path)
            add(registry, "skill", name, description, body, f"{tool}:{path.parent.name}", "active")
    for path in ARCHIVE_ROOT.rglob("SKILL.md") if ARCHIVE_ROOT.exists() else []:
        name, description, body = read_skill(path)
        add(registry, "skill", name, description, body, f"archive:{path.parent.name}", "archived")
    for path in PLUGIN_CACHE.rglob("skills/*/SKILL.md") if PLUGIN_CACHE.exists() else []:
        name, description, body = read_skill(path)
        add(registry, "skill", name, description, body, f"plugin:{path.parent.name}", "active")


def scan_agents(registry: dict[str, Capability]) -> None:
    for path in AGENTS_DIR.glob("*.md") if AGENTS_DIR.exists() else []:
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8", errors="ignore"))
        add(registry, "agent", meta.get("name", path.stem), meta.get("description", ""), body.strip(),
            f"claude-agent:{path.name}", "active")


def scan_mcp_servers(registry: dict[str, Capability]) -> None:
    if not CLAUDE_JSON.exists():
        return
    servers = json.loads(CLAUDE_JSON.read_text(encoding="utf-8")).get("mcpServers", {})
    for name, cfg in servers.items():
        command = " ".join([str(cfg.get("command", ""))] + [str(a) for a in cfg.get("args", [])])
        add(registry, "mcp_server", name, f"MCP server `{name}` ({command[:120]})", "", f"claude.json:{name}", "active")


def scan_context(registry: dict[str, Capability]) -> None:
    """One capability per top-level section of each context file, so chunks can be chosen separately."""
    for path in CONTEXT_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for section in re.split(r"(?m)^(?=## )", text):
            title = section.splitlines()[0].lstrip("# ").strip() if section.strip() else ""
            if title:
                add(registry, "context", f"{path.name}: {title}", title, section.strip(), str(path), "active")


def build() -> dict[str, Capability]:
    registry: dict[str, Capability] = {}
    scan_skills(registry)
    scan_agents(registry)
    scan_mcp_servers(registry)
    scan_context(registry)
    return registry


def summarise(registry: dict[str, Capability]) -> str:
    rows: dict[tuple[str, str], int] = {}
    for cap in registry.values():
        rows[(cap.kind, cap.status)] = rows.get((cap.kind, cap.status), 0) + 1
    active_listing = sum(c.tokens_description for c in registry.values() if c.kind == "skill" and c.status == "active")
    lines = [f"{kind:11} {status:9} {count}" for (kind, status), count in sorted(rows.items())]
    lines.append(f"active skill descriptions: ~{active_listing} tokens if all listed")
    lines.append(f"skills present in 2+ places: {sum(1 for c in registry.values() if c.kind == 'skill' and len(c.locations) > 1)}")
    return "\n".join(lines)


def main() -> None:
    registry = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps([asdict(c) for c in registry.values()], indent=1), encoding="utf-8")
    print(summarise(registry))
    print(f"wrote {OUT_PATH} ({len(registry)} capabilities)")


if __name__ == "__main__":
    sys.exit(main())
