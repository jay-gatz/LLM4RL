import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


BIB_ENTRY_RE = re.compile(r"(?m)^[ \t]*@\w+\s*\{\s*([^,\s]+)\s*,")
CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(input|include)\{([^}]+)\}")
BEGIN_ENV_RE = re.compile(r"\\begin\{([^}]+)\}")
END_ENV_RE = re.compile(r"\\end\{([^}]+)\}")

SECTION_RE = re.compile(r"\\section\{([^}]*)\}")
SUBSECTION_RE = re.compile(r"\\subsection\{([^}]*)\}")

TABLE_ENVS = {
    "table",
    "table*",
    "longtable",
    "sidewaystable",
    "sidewaystable*",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_bibkeys(bib_path: Path) -> List[str]:
    text = _read_text(bib_path)
    keys = BIB_ENTRY_RE.findall(text)
    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _resolve_input_path(cur_file: Path, rel: str) -> Optional[Path]:
    rel = rel.strip()
    if not rel:
        return None
    cand = (cur_file.parent / rel)
    if cand.suffix == "":
        cand = cand.with_suffix(".tex")
    if cand.exists():
        return cand
    raw = (cur_file.parent / rel)
    if raw.exists():
        return raw
    return None


def expand_tex(root_tex: Path, max_depth: int = 20) -> str:
    """
    Recursively expand \\input/\\include in-place so we can attribute citations
    to the subsection they appear under.
    """
    root_tex = root_tex.resolve()
    visiting: Set[Path] = set()

    def _expand_one(p: Path, depth: int) -> str:
        if depth > max_depth:
            return ""
        p = p.resolve()
        if p in visiting or not p.exists():
            return ""
        visiting.add(p)
        text = _read_text(p)

        out_parts: List[str] = []
        last = 0
        for m in INPUT_RE.finditer(text):
            out_parts.append(text[last : m.start()])
            child = _resolve_input_path(p, m.group(2))
            if child is not None:
                out_parts.append(f"\n% BEGIN_INPUT {child.name}\n")
                out_parts.append(_expand_one(child, depth + 1))
                out_parts.append(f"\n% END_INPUT {child.name}\n")
            else:
                # Keep the original command if we can't resolve it.
                out_parts.append(m.group(0))
            last = m.end()
        out_parts.append(text[last:])
        visiting.remove(p)
        return "".join(out_parts)

    return _expand_one(root_tex, 0)


@dataclass
class SubsectionStats:
    section: str
    subsection: str
    keys_text: Set[str]
    keys_table: Set[str]
    cite_cmds_text: int
    cite_cmds_table: int


def compute_subsection_citation_stats(expanded_tex: str, bibset: Set[str]) -> List[SubsectionStats]:
    cur_section = "(none)"
    cur_subsection = "(preamble)"
    env_stack: List[str] = []

    stats_map: Dict[Tuple[str, str], SubsectionStats] = {}

    def _get_stats() -> SubsectionStats:
        key = (cur_section, cur_subsection)
        if key not in stats_map:
            stats_map[key] = SubsectionStats(
                section=cur_section,
                subsection=cur_subsection,
                keys_text=set(),
                keys_table=set(),
                cite_cmds_text=0,
                cite_cmds_table=0,
            )
        return stats_map[key]

    for line in expanded_tex.splitlines():
        sm = SECTION_RE.search(line)
        if sm:
            cur_section = sm.group(1).strip()
            cur_subsection = "(none)"
        ssm = SUBSECTION_RE.search(line)
        if ssm:
            cur_subsection = ssm.group(1).strip()

        for bm in BEGIN_ENV_RE.finditer(line):
            env_stack.append(bm.group(1))
        for em in END_ENV_RE.finditer(line):
            if env_stack and env_stack[-1] == em.group(1):
                env_stack.pop()
            else:
                if em.group(1) in env_stack:
                    while env_stack and env_stack[-1] != em.group(1):
                        env_stack.pop()
                    if env_stack and env_stack[-1] == em.group(1):
                        env_stack.pop()

        in_table = any(e in TABLE_ENVS for e in env_stack)
        for cm in CITE_RE.finditer(line):
            raw = cm.group(1)
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            s = _get_stats()
            if in_table:
                s.cite_cmds_table += 1
                for k in keys:
                    if k in bibset:
                        s.keys_table.add(k)
            else:
                s.cite_cmds_text += 1
                for k in keys:
                    if k in bibset:
                        s.keys_text.add(k)

    # Deterministic ordering (as it appears in the manuscript).
    ordered: List[SubsectionStats] = []
    seen = set()
    for line in expanded_tex.splitlines():
        sm = SECTION_RE.search(line)
        if sm:
            cur_section = sm.group(1).strip()
            cur_subsection = "(none)"
        ssm = SUBSECTION_RE.search(line)
        if ssm:
            cur_subsection = ssm.group(1).strip()
        key = (cur_section, cur_subsection)
        if key in stats_map and key not in seen:
            ordered.append(stats_map[key])
            seen.add(key)

    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default=".", help="Project root containing tex/bib.")
    ap.add_argument("--tex", type=str, default="sn-article.tex", help="Root TeX file path (relative to project).")
    ap.add_argument("--bib", type=str, default="sn-bibliography.bib", help="Bib file path (relative to project).")
    ap.add_argument("--out", type=str, default="citation_density_report_internal.md", help="Markdown output path.")
    ap.add_argument("--min", type=int, default=8, help="Minimum unique in-text citations per subsection.")
    args = ap.parse_args()

    project_root = Path(args.project).resolve()
    root_tex = (project_root / args.tex).resolve()
    bib_path = (project_root / args.bib).resolve()
    if not root_tex.exists():
        raise SystemExit(f"Missing root tex: {root_tex}")
    if not bib_path.exists():
        raise SystemExit(f"Missing bib: {bib_path}")

    bibkeys = extract_bibkeys(bib_path)
    bibset = set(bibkeys)

    expanded = expand_tex(root_tex)
    stats = compute_subsection_citation_stats(expanded, bibset)

    # Filter to "real" subsections only (ignore preamble and (none)).
    real = [s for s in stats if s.subsection not in ("(preamble)", "(none)")]
    low = [s for s in real if len(s.keys_text) < args.min]

    lines: List[str] = []
    lines.append("# Citation Density Report (Internal)")
    lines.append("")
    lines.append(f"- Total bib entries: **{len(bibkeys)}**")
    lines.append(f"- Subsections analyzed: **{len(real)}**")
    lines.append(f"- Subsections below min={args.min} (unique in-text keys): **{len(low)}**")
    lines.append("")
    lines.append("## Per-Subsection Counts")
    lines.append("")
    lines.append("| Section | Subsection | Unique In-Text Keys | Cite Cmds (Text) | Unique In-Table Keys | Cite Cmds (Table) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for s in real:
        lines.append(
            f"| {s.section} | {s.subsection} | {len(s.keys_text)} | {s.cite_cmds_text} | {len(s.keys_table)} | {s.cite_cmds_table} |"
        )
    lines.append("")
    lines.append(f"## Below Threshold (< {args.min} Unique In-Text Keys)")
    lines.append("")
    if not low:
        lines.append("None.")
    else:
        for s in low:
            lines.append(f"- **{s.section} / {s.subsection}**: {len(s.keys_text)} unique in-text keys")

    out_path = (project_root / args.out).resolve()
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    # Print a compact summary for iteration loops.
    print(f"SUBSECTIONS_ANALYZED={len(real)}")
    print(f"SUBSECTIONS_BELOW_MIN={len(low)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

