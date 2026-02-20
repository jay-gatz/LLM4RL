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


TABLE_ENVS = {
    "table",
    "table*",
    "longtable",
    "sidewaystable",
    "sidewaystable*",
}


@dataclass
class CiteOccurrence:
    key: str
    in_table: bool
    in_appendix: bool
    source: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_bibkeys(bib_path: Path) -> List[str]:
    text = _read_text(bib_path)
    keys = BIB_ENTRY_RE.findall(text)
    # Preserve deterministic ordering (as in the bib file).
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def resolve_inputs(root_tex: Path, max_depth: int = 20) -> List[Path]:
    """
    Resolve \\input and \\include recursively starting from root_tex.
    Returns a deterministic list of source files (including root_tex).
    """
    root_tex = root_tex.resolve()
    seen: Set[Path] = set()
    ordered: List[Path] = []

    def _resolve_one(p: Path, depth: int) -> None:
        if depth > max_depth:
            return
        p = p.resolve()
        if p in seen or not p.exists():
            return
        seen.add(p)
        ordered.append(p)
        text = _read_text(p)
        for m in INPUT_RE.finditer(text):
            rel = m.group(2).strip()
            if not rel:
                continue
            # LaTeX resolves relative to the current file directory.
            cand = (p.parent / rel)
            if cand.suffix == "":
                cand = cand.with_suffix(".tex")
            # We still include .tikz/.tex figure sources to be safe.
            if cand.exists():
                _resolve_one(cand, depth + 1)
            else:
                # Try raw path (some inputs omit .tex intentionally).
                if (p.parent / rel).exists():
                    _resolve_one((p.parent / rel), depth + 1)

    _resolve_one(root_tex, 0)
    return ordered


def extract_citations_with_context(files: Iterable[Path], project_root: Path) -> List[CiteOccurrence]:
    occs: List[CiteOccurrence] = []
    # We approximate appendix boundary by the first occurrence of \\appendix in the root manuscript.
    # For included files, we treat them as "appendix" if they are included after \\appendix appears
    # in the root file. We implement this by tracking a global flag while parsing the root file,
    # and for other files we default to "unknown" but assume they are included in-place (good enough).
    # For strictness, we detect \\appendix in each file as well.

    for p in files:
        text = _read_text(p)
        env_stack: List[str] = []
        in_appendix = False
        # A simple line-based parser to track env nesting.
        for line in text.splitlines():
            if "\\appendix" in line:
                in_appendix = True
            for bm in BEGIN_ENV_RE.finditer(line):
                env_stack.append(bm.group(1))
            for em in END_ENV_RE.finditer(line):
                if env_stack and env_stack[-1] == em.group(1):
                    env_stack.pop()
                else:
                    # Best-effort: pop until match if present.
                    if em.group(1) in env_stack:
                        while env_stack and env_stack[-1] != em.group(1):
                            env_stack.pop()
                        if env_stack and env_stack[-1] == em.group(1):
                            env_stack.pop()

            in_table = any(e in TABLE_ENVS for e in env_stack)
            for cm in CITE_RE.finditer(line):
                raw = cm.group(1)
                for k in raw.split(","):
                    kk = k.strip()
                    if not kk:
                        continue
                    occs.append(
                        CiteOccurrence(
                            key=kk,
                            in_table=in_table,
                            in_appendix=in_appendix,
                            source=str(p.relative_to(project_root)).replace("\\", "/"),
                        )
                    )
    return occs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default=".", help="Project root containing the tex/bib.")
    ap.add_argument("--tex", type=str, default="sn-article.tex", help="Root TeX file path (relative to project).")
    ap.add_argument("--bib", type=str, default="sn-bibliography.bib", help="Bib file path (relative to project).")
    ap.add_argument("--out", type=str, default="coverage_integrity_report_internal.md", help="Markdown report output.")
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

    files = resolve_inputs(root_tex)
    occs = extract_citations_with_context(files, project_root)

    cited_any = {o.key for o in occs}
    cited_text = {o.key for o in occs if not o.in_table}
    cited_tables = {o.key for o in occs if o.in_table}

    # Appendix-only means it never occurs before appendix marker in the root file. Since our parser
    # marks appendix per-file, we compute a weaker proxy: occurs only in lines after \\appendix in that file.
    # This is still useful for auditing.
    cited_main_any = {o.key for o in occs if not o.in_appendix}
    cited_appendix_any = {o.key for o in occs if o.in_appendix}

    not_cited_text = sorted(bibset - cited_text)
    not_in_table = sorted(bibset - cited_tables)
    only_appendix = sorted((cited_appendix_any - cited_main_any) & bibset)
    only_refs = sorted(bibset - cited_any)

    report_lines: List[str] = []
    report_lines.append("# Coverage Integrity Report (Internal)")
    report_lines.append("")
    report_lines.append(f"- Total bibliography entries: **{len(bibkeys)}**")
    report_lines.append(f"- Cited anywhere (tex): **{len(cited_any & bibset)}**")
    report_lines.append(f"- Cited in text (non-table): **{len(cited_text & bibset)}**")
    report_lines.append(f"- Appearing in tables: **{len(cited_tables & bibset)}**")
    report_lines.append(f"- Appendix-only (proxy): **{len(only_appendix)}**")
    report_lines.append(f"- Only in references (orphan): **{len(only_refs)}**")
    report_lines.append("")

    def _emit_list(title: str, keys: List[str], limit: int = 200) -> None:
        report_lines.append(f"## {title} ({len(keys)})")
        report_lines.append("")
        if not keys:
            report_lines.append("None.")
            report_lines.append("")
            return
        if len(keys) > limit:
            shown = keys[:limit]
            report_lines.append("Shown (first {}):".format(limit))
            report_lines.append("")
            report_lines.append("```")
            report_lines.extend(shown)
            report_lines.append("```")
            report_lines.append("")
            report_lines.append(f"(Truncated; total {len(keys)}.)")
            report_lines.append("")
            return
        report_lines.append("```")
        report_lines.extend(keys)
        report_lines.append("```")
        report_lines.append("")

    _emit_list("Papers Not Cited In Text (Non-Table)", not_cited_text)
    _emit_list("Papers Not Appearing In Any Table", not_in_table)
    _emit_list("Papers Appearing Only In Appendix (Proxy)", only_appendix)
    _emit_list("Papers Appearing Only In References (Orphans)", only_refs)

    # Write report.
    out_path = (project_root / args.out).resolve()
    out_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    # Also print final summary to stdout for iteration automation.
    print(f"TOTAL_BIB={len(bibkeys)}")
    print(f"CITED_TEXT={len(cited_text & bibset)}")
    print(f"CITED_TABLES={len(cited_tables & bibset)}")
    print(f"NOT_CITED_TEXT={len(not_cited_text)}")
    print(f"NOT_IN_TABLE={len(not_in_table)}")
    print(f"ONLY_REFS={len(only_refs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

