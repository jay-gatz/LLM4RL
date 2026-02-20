import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

# This generator rewrites the Springer Nature template manuscript (paper/sn-article.tex)
# from the verified bibliography in core_llm_for_rl_v3.csv. It intentionally performs
# no web searches: all figures/tables/stats are derived from the CSV only.

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent

CORE_CSV = REPO_ROOT / "core_llm_for_rl_v3.csv"

TEX_OUT = PAPER_DIR / "sn-article.tex"
BIB_OUT = PAPER_DIR / "sn-bibliography.bib"

CITATION_REPORT = PAPER_DIR / "citation_coverage_report.md"


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def latex_escape_text(s: str) -> str:
    """Escape plain text for LaTeX. Do NOT use on strings that already contain LaTeX."""
    s = clean_space(s)
    repl = {
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
    return "".join(repl.get(ch, ch) for ch in s)


def sanitize_key(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", clean_space(s))
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "paper"
    if not re.match(r"^[A-Za-z]", s):
        s = "p_" + s
    return s


def normalize_canonical_id(raw: str) -> str:
    """
    Normalize the CSV ArXiv_or_DOI field into either:
      - arXiv:<id>   (no version suffix)
      - <doi>
    """
    v = clean_space(raw)
    if not v:
        return v

    # Normalize arXiv URL -> arXiv:ID
    v = re.sub(r"^https?://arxiv\.org/abs/", "arXiv:", v, flags=re.IGNORECASE)

    # Normalize arXiv DOI form -> arXiv:ID
    m_arxiv_doi = re.match(r"^10\.48550/arxiv\.(\d{4}\.\d{4,5})(v\d+)?$", v, flags=re.IGNORECASE)
    if m_arxiv_doi:
        v = f"arXiv:{m_arxiv_doi.group(1)}"

    # Strip arXiv version suffix
    v = re.sub(r"^(arXiv:[0-9.]+)v\d+$", r"\1", v, flags=re.IGNORECASE)
    return v


def canonical_id_to_bibkey(canonical_id: str) -> str:
    cid = clean_space(canonical_id)
    if cid.lower().startswith("arxiv:"):
        arx = cid.split(":", 1)[1].strip()
        return "arxiv_" + sanitize_key(arx)
    return "doi_" + sanitize_key(cid.lower())


def split_authors(authors: str) -> List[str]:
    a = clean_space(authors)
    if not a:
        return []
    # CSV uses comma-separated author lists; keep "and" if present.
    if " and " in a:
        parts = [p.strip() for p in a.split(" and ") if p.strip()]
    else:
        parts = [p.strip() for p in a.split(",") if p.strip()]
    return parts


def bib_escape(s: str) -> str:
    """BibTeX field escape (conservative)."""
    s = clean_space(s)
    s = s.replace("\\", " ")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    return s


def tag_set(tag_str: str) -> Set[str]:
    return {t.strip() for t in clean_space(tag_str).split("|") if t.strip()}


MAJOR_AXES = [
    "Information Processor",
    "Planner & Task Decomposer",
    "Policy / Program Generator",
    "Reward Designer & Verifier",
    "World Model / Simulator / Imagination",
    "Exploration / Curriculum / Goal Generation",
    "Offline RL Data & Trajectory Augmentation",
    "Tool & Memory Interfaces",
    "Search Hybrids",
]


def major_axes_for_tags(tags: Set[str]) -> List[str]:
    out: List[str] = []
    if "Reasoning" in tags or "LanguageConditioning" in tags:
        out.append("Information Processor")
    if "Planning" in tags:
        out.append("Planner & Task Decomposer")
    if "Policy" in tags:
        out.append("Policy / Program Generator")
    if ("Reward" in tags) or ("Verifier" in tags) or ("CreditAssignment" in tags):
        out.append("Reward Designer & Verifier")
    if ("WorldModel" in tags) or ("Simulation" in tags):
        out.append("World Model / Simulator / Imagination")
    if ("Exploration" in tags) or ("Curriculum" in tags):
        out.append("Exploration / Curriculum / Goal Generation")
    if ("OfflineRL" in tags) or ("DatasetGeneration" in tags):
        out.append("Offline RL Data & Trajectory Augmentation")
    if ("ToolUse" in tags) or ("Memory" in tags):
        out.append("Tool & Memory Interfaces")
    if "TreeSearch" in tags:
        out.append("Search Hybrids")
    return out


def primary_axis(tags: Set[str]) -> str:
    """
    Choose a single primary mechanism axis for compact tables.
    Deterministic precedence is used to avoid ambiguity.
    """
    if "TreeSearch" in tags:
        return "Search Hybrids"
    if ("Reward" in tags) or ("Verifier" in tags) or ("CreditAssignment" in tags):
        return "Reward Designer & Verifier"
    if ("WorldModel" in tags) or ("Simulation" in tags):
        return "World Model / Simulator / Imagination"
    if "Planning" in tags:
        return "Planner & Task Decomposer"
    if ("ToolUse" in tags) or ("Memory" in tags):
        return "Tool & Memory Interfaces"
    if ("OfflineRL" in tags) or ("DatasetGeneration" in tags):
        return "Offline RL Data & Trajectory Augmentation"
    if ("Exploration" in tags) or ("Curriculum" in tags):
        return "Exploration / Curriculum / Goal Generation"
    if "Policy" in tags:
        return "Policy / Program Generator"
    if "Reasoning" in tags or "LanguageConditioning" in tags:
        return "Information Processor"
    return "Policy / Program Generator"


def rl_setting_from_tags(tags: Set[str]) -> str:
    if ("OfflineRL" in tags) or ("DatasetGeneration" in tags):
        return "Offline/Batch RL"
    if ("WorldModel" in tags) or ("Simulation" in tags):
        return "Model-based RL"
    if "TreeSearch" in tags:
        return "Search-augmented"
    if "Planning" in tags:
        return "Hierarchical/Long-horizon"
    if ("Exploration" in tags) or ("Curriculum" in tags):
        return "Exploration/Curriculum"
    if "MultiAgent" in tags:
        return "Multi-agent RL"
    return "General/Online RL"


BENCH_TOKENS = [
    "Minecraft",
    "WebShop",
    "ALFWorld",
    "TextWorld",
    "Atari",
    "MuJoCo",
    "D4RL",
    "MiniHack",
    "Procgen",
    "BabyAI",
    "Overcooked",
    "Crafter",
]


def domain_from_summary(title: str, summary: str) -> str:
    txt = (title + " " + summary).lower()
    if "multi-agent" in txt or "multi agent" in txt:
        return "Multi-Agent"
    if any(
        k in txt
        for k in [
            "webshop",
            "alfworld",
            "browser",
            "website",
            "web agent",
            "web/",
            "web ",
            "gui",
            "navigation",
        ]
    ):
        return "Web/Interactive"
    if any(k in txt for k in ["minecraft", "textworld", "atari", "game", "games", "text environment"]):
        return "Games/Text"
    if any(k in txt for k in ["mujoco", "continuous control", "locomotion"]):
        return "Continuous Control"
    if any(k in txt for k in ["robot", "robotics", "embodied", "manipulation"]):
        return "Robotics/Embodied"
    return "General/Unspecified"


AXIS_ABBREV = {
    "Information Processor": "InfoProc",
    "Planner & Task Decomposer": "Planner",
    "Policy / Program Generator": "Policy",
    "Reward Designer & Verifier": "RwdVer",
    "World Model / Simulator / Imagination": "WorldSim",
    "Exploration / Curriculum / Goal Generation": "Explore",
    "Offline RL Data & Trajectory Augmentation": "OffData",
    "Tool & Memory Interfaces": "ToolMem",
    "Search Hybrids": "Search",
}

SETTING_ABBREV = {
    "Offline/Batch RL": "Offline",
    "Model-based RL": "Model",
    "Search-augmented": "Search",
    "Hierarchical/Long-horizon": "Hier",
    "Exploration/Curriculum": "Explore",
    "Multi-agent RL": "Multi-agent",
    "General/Online RL": "Online",
}

DOMAIN_ABBREV = {
    "Robotics/Embodied": "Robot",
    "Web/Interactive": "Web",
    "Games/Text": "Games",
    "Continuous Control": "Cont",
    "Multi-Agent": "MultiAg",
    "General/Unspecified": "General",
}


def abbrev_axis(axis: str) -> str:
    return AXIS_ABBREV.get(axis, axis)


def abbrev_setting(setting: str) -> str:
    return SETTING_ABBREV.get(setting, setting)


def abbrev_domain(domain: str) -> str:
    return DOMAIN_ABBREV.get(domain, domain)


def extract_benchmarks(summary: str) -> str:
    s = summary or ""
    found = []
    for tok in BENCH_TOKENS:
        if re.search(r"\b" + re.escape(tok) + r"\b", s, flags=re.IGNORECASE):
            found.append(tok)
    return ", ".join(found)


def cleaned_contribution_for_appendix(title: str, summary: str) -> str:
    """
    Shorten a CSV summary for appendix usage without adding new facts.
    """
    s = clean_space(summary)
    t = clean_space(title)
    patterns = [
        rf"^In\s+{re.escape(t)}\s+",
        rf"^For\s+{re.escape(t)},\s*",
        rf"^{re.escape(t)}\s+",
    ]
    for pat in patterns:
        s2 = re.sub(pat, "", s, flags=re.IGNORECASE)
        if s2 != s:
            s = s2.strip()
            break
    # Remove common boilerplate openings.
    s = re.sub(r"^(In this work,|This work|We)\s+", "", s, flags=re.IGNORECASE)

    # Keep complete sentences; drop evaluation/metric-only sentences to avoid overclaiming.
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]
    bad = re.compile(
        r"(Evaluation is conducted|Results are reported|Empirical results|Experiments on|success rate|returns|sample efficiency|improvements?)",
        flags=re.IGNORECASE,
    )
    kept = [x for x in sentences if not bad.search(x)]
    if not kept:
        kept = sentences

    out = ""
    for sent in kept:
        cand = sent if not out else (out + " " + sent)
        if len(cand) > 260:
            break
        out = cand
        if len(out) >= 170:
            break

    out = out.strip()
    if out and out[-1] not in ".!?":
        out += "."
    if len(out) > 260:
        # Fall back to a safe word-boundary cut.
        cut = out[:260]
        cut = re.sub(r"\s+\S*$", "", cut)
        out = cut.rstrip(".") + "."
    return out


def select_representatives_by_filter(
    df: pd.DataFrame,
    mask: pd.Series,
    n: int,
    must: Optional[List[str]] = None,
) -> List[str]:
    must = must or []
    sub = df[mask].copy()
    sub = sub.sort_values(["Year", "Title"], ascending=[False, True])
    out: List[str] = []
    for k in must:
        if k and k not in out:
            out.append(k)
    for k in sub["BibKey"].tolist():
        if k not in out:
            out.append(k)
        if len(out) >= n:
            break
    return out[:n]


def cite(keys: Iterable[str], bucket: Set[str]) -> str:
    ks = [k for k in keys if k]
    bucket.update(ks)
    return r"\cite{" + ",".join(ks) + r"}" if ks else ""


def cite_grouped(keys: Iterable[str], bucket: Set[str], group_size: int = 4) -> str:
    """
    Emit multiple \\cite{...} groups to keep long citation lists breakable in narrow tables.
    """
    ks = [k for k in keys if k]
    bucket.update(ks)
    if not ks:
        return ""
    groups = [ks[i : i + group_size] for i in range(0, len(ks), group_size)]
    return "; ".join(r"\cite{" + ",".join(g) + r"}" for g in groups)


def ensure_bst_present() -> None:
    # BibTeX expects the .bst to be in the working dir.
    bst_local = PAPER_DIR / "sn-mathphys-num.bst"
    if bst_local.exists():
        return
    bst_src = PAPER_DIR / "bst" / "sn-mathphys-num.bst"
    if bst_src.exists():
        shutil.copyfile(bst_src, bst_local)


def write_bib(df: pd.DataFrame) -> None:
    out: List[str] = []
    for _, r in df.iterrows():
        key = r["BibKey"]
        title = bib_escape(r["Title"])
        year = str(int(r["Year"])) if str(r["Year"]).strip().isdigit() else clean_space(r["Year"])
        venue = bib_escape(r["Venue"]) or "arXiv"
        cid = clean_space(r["CanonicalID"])

        authors = split_authors(r["Authors"])
        author_field = " and ".join(bib_escape(a) for a in authors) if authors else "Unknown"

        if cid.lower().startswith("arxiv:"):
            arx = cid.split(":", 1)[1].strip()
            url = f"https://arxiv.org/abs/{arx}"
            entry = [
                f"@misc{{{key},",
                f"  title = {{{title}}},",
                f"  author = {{{author_field}}},",
                f"  year = {{{year}}},",
                f"  note = {{{venue}}},",
                f"  eprint = {{{arx}}},",
                "  archivePrefix = {arXiv},",
                f"  url = {{{url}}},",
                "}",
            ]
        else:
            doi = cid
            url = f"https://doi.org/{doi}"
            entry = [
                f"@misc{{{key},",
                f"  title = {{{title}}},",
                f"  author = {{{author_field}}},",
                f"  year = {{{year}}},",
                f"  note = {{{venue}}},",
                f"  doi = {{{doi}}},",
                f"  url = {{{url}}},",
                "}",
            ]
        out.append("\n".join(entry))
        out.append("")

    BIB_OUT.write_text("\n".join(out), encoding="utf-8")


def make_table_taxonomy_overview(df: pd.DataFrame, cites_tables: Set[str], keymap: Dict[str, str]) -> str:
    # Must-haves (ensure present in representative citations)
    saycan = keymap.get("arXiv:2204.01691", "")
    cap = keymap.get("10.1109/icra48891.2023.10160591", "")
    voyager = keymap.get("arXiv:2305.16291", "")
    eureka = keymap.get("arXiv:2310.12931", "")
    text2reward = keymap.get("arXiv:2309.11489", "")

    tags = df["Category_Tags"].astype(str)
    t1_rows: List[Tuple[str, str, str, str]] = []

    def reps_for(axis_name: str, n: int) -> str:
        if axis_name == "Information Processor":
            mask = tags.str.contains("Reasoning", regex=False) | tags.str.contains("LanguageConditioning", regex=False)
            reps = select_representatives_by_filter(df, mask, n)
        elif axis_name == "Planner & Task Decomposer":
            mask = tags.str.contains("Planning", regex=False)
            reps = select_representatives_by_filter(df, mask, n, must=[saycan, voyager])
        elif axis_name == "Policy / Program Generator":
            mask = tags.str.contains("Policy", regex=False)
            reps = select_representatives_by_filter(df, mask, n, must=[cap, saycan])
        elif axis_name == "Reward Designer & Verifier":
            mask = (
                tags.str.contains("Reward", regex=False)
                | tags.str.contains("Verifier", regex=False)
                | tags.str.contains("CreditAssignment", regex=False)
            )
            reps = select_representatives_by_filter(df, mask, n, must=[eureka, text2reward])
        elif axis_name == "World Model / Simulator / Imagination":
            mask = tags.str.contains("WorldModel", regex=False) | tags.str.contains("Simulation", regex=False)
            reps = select_representatives_by_filter(df, mask, n)
        elif axis_name == "Exploration / Curriculum / Goal Generation":
            mask = tags.str.contains("Exploration", regex=False) | tags.str.contains("Curriculum", regex=False)
            reps = select_representatives_by_filter(df, mask, n, must=[voyager])
        elif axis_name == "Offline RL Data & Trajectory Augmentation":
            mask = tags.str.contains("OfflineRL", regex=False) | tags.str.contains("DatasetGeneration", regex=False)
            reps = select_representatives_by_filter(df, mask, n)
        elif axis_name == "Tool & Memory Interfaces":
            mask = tags.str.contains("ToolUse", regex=False) | tags.str.contains("Memory", regex=False)
            reps = select_representatives_by_filter(df, mask, n, must=[voyager])
        elif axis_name == "Search Hybrids":
            mask = tags.str.contains("TreeSearch", regex=False)
            reps = select_representatives_by_filter(df, mask, n)
        else:
            reps = []
        return cite_grouped(reps, cites_tables, group_size=4)

    t1_rows.append(
        (
            "Information Processor",
            "Transforms language goals/specifications into state, constraints, or latent task representations used by RL updates.",
            "Language-conditioned RL; representation learning; structured interfaces",
            reps_for("Information Processor", 8),
        )
    )
    t1_rows.append(
        (
            r"Planner \& Task Decomposer",
            "Produces subgoals/options/program sketches to reduce effective horizon; RL executes and replans under feedback.",
            "Hierarchical and long-horizon RL; embodied and web agents",
            reps_for("Planner & Task Decomposer", 8),
        )
    )
    t1_rows.append(
        (
            "Policy / Program Generator",
            "Generates executable policies, code, or action proposals that are grounded by RL values, critics, or environment rollouts.",
            "Interactive control; programmatic policies; recovery-driven agents",
            reps_for("Policy / Program Generator", 8),
        )
    )
    t1_rows.append(
        (
            r"Reward Designer \& Verifier",
            "Densifies supervision via shaped rewards, process checks, or verifiers that gate updates and filter trajectories.",
            "Sparse-reward RL; constraint satisfaction; trajectory-level supervision",
            reps_for("Reward Designer & Verifier", 8),
        )
    )
    t1_rows.append(
        (
            "World Model / Simulator",
            "Provides imagination or surrogate dynamics via language-grounded prediction; RL learns from real and imagined experience.",
            "Model-based RL; simulator-in-the-loop training; uncertainty-aware planning",
            reps_for("World Model / Simulator / Imagination", 8),
        )
    )
    t1_rows.append(
        (
            "Exploration / Curriculum",
            "Generates goals/tasks and adapts curricula to competence to improve exploration and coverage.",
            "Goal-conditioned RL; curriculum RL; open-ended skill discovery",
            reps_for("Exploration / Curriculum / Goal Generation", 8),
        )
    )
    t1_rows.append(
        (
            "Offline Data Augmentation",
            "Synthesizes, annotates, or relabels trajectories to improve offline/batch RL and reduce data bottlenecks.",
            "Offline RL; dataset synthesis; trajectory relabeling",
            reps_for("Offline RL Data & Trajectory Augmentation", 8),
        )
    )
    t1_rows.append(
        (
            r"Tool \& Memory Interfaces",
            "Uses tools and memory as part of the agent state; LLM mediates queries/actions while RL grounds long-horizon control.",
            "Tool-augmented RL; multi-turn agents; state construction via memory",
            reps_for("Tool & Memory Interfaces", 8),
        )
    )
    t1_rows.append(
        (
            "Search Hybrids",
            "Combines tree search with LLM priors and RL value grounding to control branching and evaluate rollouts.",
            "MCTS/tree search; value-guided expansion; compute-aware planning",
            reps_for("Search Hybrids", 8),
        )
    )

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Leave enough width for representative citation lists in the last column.
    lines.append(r"\begin{tabularx}{\textwidth}{p{2.3cm}p{4.8cm}p{2.8cm}Y}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Axis} & \textbf{Mechanism (What the LLM changes)} & \textbf{Typical RL setting} & \textbf{Representative works} \\")
    lines.append(r"\midrule")
    for a, b, c, d in t1_rows:
        lines.append(f"{a} & {latex_escape_text(b)} & {latex_escape_text(c)} & {d} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(
        r"\caption{Taxonomy overview of LLM-to-RL mechanisms. Each row identifies where the language module intervenes in the RL loop and cites representative studies.}"
    )
    lines.append(r"\label{tab:T1}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_table_mechanism_setting_domain(df: pd.DataFrame, cites_tables: Set[str]) -> str:
    # Derived labels are computed only from CSV tags/summaries (no external sources).
    df2 = df.copy()
    df2["PrimaryAxis"] = [primary_axis(tag_set(s)) for s in df2["Category_Tags"].tolist()]
    df2["RLSetting"] = [rl_setting_from_tags(tag_set(s)) for s in df2["Category_Tags"].tolist()]
    df2["Domain"] = [domain_from_summary(t, s) for t, s in zip(df2["Title"], df2["Contribution_Summary"])]

    # For each axis: show top-2 RL settings and domains with counts + 3 example citations.
    axes = MAJOR_AXES
    rows: List[Tuple[str, str, str, str]] = []
    for ax in axes:
        sub = df2[df2["PrimaryAxis"] == ax]
        if len(sub) == 0:
            continue
        top_settings = sub["RLSetting"].value_counts().head(2).to_dict()
        top_domains = sub["Domain"].value_counts().head(2).to_dict()
        setting_str = "; ".join(f"{k} ({v})" for k, v in top_settings.items())
        domain_str = "; ".join(f"{k} ({v})" for k, v in top_domains.items())
        reps = sub.sort_values(["Year", "Title"], ascending=[False, True])["BibKey"].head(3).tolist()
        rows.append((ax, setting_str, domain_str, cite(reps, cites_tables)))

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Keep the examples column wide enough to avoid overflow.
    lines.append(r"\begin{tabularx}{\textwidth}{p{3.0cm}p{4.0cm}p{3.6cm}Y}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Axis} & \textbf{Top RL settings} & \textbf{Top domains} & \textbf{Examples} \\")
    lines.append(r"\midrule")
    for a, b, c, d in rows:
        lines.append(f"{latex_escape_text(a)} & {latex_escape_text(b)} & {latex_escape_text(c)} & {d} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(
        r"\caption{Mechanism-by-setting-by-domain summary across the surveyed literature. Counts report how often each mechanism appears with the most frequent RL settings and domains.}"
    )
    lines.append(r"\label{tab:T2}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_table_domain_category_matrix(df: pd.DataFrame) -> str:
    # Domain is derived from CSV summaries. Categories are the major axes.
    df2 = df.copy()
    df2["Domain"] = [domain_from_summary(t, s) for t, s in zip(df2["Title"], df2["Contribution_Summary"])]
    df2["PrimaryAxis"] = [primary_axis(tag_set(s)) for s in df2["Category_Tags"].tolist()]

    domains = ["Robotics/Embodied", "Web/Interactive", "Games/Text", "Continuous Control", "Multi-Agent", "General/Unspecified"]
    axes = MAJOR_AXES

    mat: Dict[Tuple[str, str], int] = {}
    for d in domains:
        for a in axes:
            mat[(d, a)] = int(((df2["Domain"] == d) & (df2["PrimaryAxis"] == a)).sum())

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{l" + "r" * len(axes) + r"}")
    lines.append(r"\toprule")
    lines.append("Domain & " + " & ".join([latex_escape_text(a.replace(" & ", "/")) for a in axes]) + r" \\")
    lines.append(r"\midrule")
    for d in domains:
        row = [latex_escape_text(d)] + [str(mat[(d, a)]) for a in axes]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(
        r"\caption{Benchmark and domain usage matrix. Entries count papers in each domain for each primary mechanism axis.}"
    )
    lines.append(r"\label{tab:T3}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_table_reward_verifier_patterns(df: pd.DataFrame) -> str:
    tags = df["Category_Tags"].astype(str)
    reward_n = int(tags.str.contains("Reward", regex=False).sum())
    ver_n = int(tags.str.contains("Verifier", regex=False).sum())
    both_n = int((tags.str.contains("Reward", regex=False) & tags.str.contains("Verifier", regex=False)).sum())
    ca_n = int((tags.str.contains("Reward", regex=False) & tags.str.contains("CreditAssignment", regex=False)).sum())

    rows = [
        ("LLM reward shaping", reward_n, "Language-derived reward terms or reward code densify sparse supervision."),
        ("Verifier supervision", ver_n, "Trajectory checks gate updates or filter rollouts to reduce reward hacking."),
        ("Reward + Verifier", both_n, "Shaping and verification are combined to improve robustness under proxy signals."),
        ("Reward + Credit assignment", ca_n, "Intermediate signals support delayed credit attribution across long horizons."),
    ]

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabularx}{\columnwidth}{p{3.2cm}rY}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Pattern} & \textbf{Count} & \textbf{Interpretation} \\")
    lines.append(r"\midrule")
    for a, b, c in rows:
        lines.append(f"{latex_escape_text(a)} & {b} & {latex_escape_text(c)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\caption{Reward and verifier design patterns in the surveyed LLM-to-RL literature.}")
    lines.append(r"\label{tab:T4}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_table_tool_memory_patterns(df: pd.DataFrame) -> str:
    tags = df["Category_Tags"].astype(str)
    tool_n = int(tags.str.contains("ToolUse", regex=False).sum())
    mem_n = int(tags.str.contains("Memory", regex=False).sum())
    tool_ver = int((tags.str.contains("ToolUse", regex=False) & tags.str.contains("Verifier", regex=False)).sum())
    tool_ma = int((tags.str.contains("ToolUse", regex=False) & tags.str.contains("MultiAgent", regex=False)).sum())

    rows = [
        ("Tool-use integrated policies", tool_n, "Agents call external tools inside episodes; tool traces become part of state."),
        ("Memory-augmented loops", mem_n, "Long-horizon summaries compress interaction history for control and learning."),
        ("Tool + Verifier co-design", tool_ver, "Verification guards tool errors and prevents compounding mistakes."),
        ("Tool + Multi-agent settings", tool_ma, "Coordination shares tool outcomes or memory state across agents."),
    ]

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabularx}{\columnwidth}{p{3.2cm}rY}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Pattern} & \textbf{Count} & \textbf{Interpretation} \\")
    lines.append(r"\midrule")
    for a, b, c in rows:
        lines.append(f"{latex_escape_text(a)} & {b} & {latex_escape_text(c)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\caption{Tool and memory integration patterns in LLM-modulated RL loops.}")
    lines.append(r"\label{tab:T5}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_table_year_category_trends(df: pd.DataFrame) -> str:
    years = sorted(int(y) for y in df["Year"].unique() if int(y) > 0)
    cols = [
        ("InfoProc", lambda ts: ("Reasoning" in ts) or ("LanguageConditioning" in ts)),
        ("Planner", lambda ts: "Planning" in ts),
        ("Policy", lambda ts: "Policy" in ts),
        ("Reward/Verifier", lambda ts: ("Reward" in ts) or ("Verifier" in ts) or ("CreditAssignment" in ts)),
        ("World/Sim", lambda ts: ("WorldModel" in ts) or ("Simulation" in ts)),
        ("Expl./Curr.", lambda ts: ("Exploration" in ts) or ("Curriculum" in ts)),
        ("Offline Data", lambda ts: ("OfflineRL" in ts) or ("DatasetGeneration" in ts)),
        ("Tool/Memory", lambda ts: ("ToolUse" in ts) or ("Memory" in ts)),
        ("Search", lambda ts: "TreeSearch" in ts),
    ]

    tag_sets = [tag_set(s) for s in df["Category_Tags"].tolist()]

    rows: List[List[str]] = []
    for y in years:
        row = [str(y)]
        idxs = df.index[df["Year"] == y].tolist()
        for _, pred in cols:
            v = 0
            for i in idxs:
                if pred(tag_sets[i]):
                    v += 1
            row.append(str(v))
        rows.append(row)

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{l" + "r" * len(cols) + r"}")
    lines.append(r"\toprule")
    lines.append("Year & " + " & ".join([latex_escape_text(n) for n, _ in cols]) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{Year-by-category trend summary across the surveyed corpus.}")
    lines.append(r"\label{tab:T6}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def tikz_figure_loop() -> str:
    # Minimal conceptual layout: core RL loop with four LLM modules.
    return r"""
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  font=\small,
  core/.style={rounded corners, draw=black, very thick, minimum width=3.1cm, minimum height=1.2cm, align=center, fill=white},
  module/.style={rounded corners, draw=black, thick, minimum width=3.2cm, minimum height=1.0cm, align=center, fill=gray!8},
  loop/.style={-Latex, very thick},
  mod/.style={-Latex, thick}
]
  \node[core] (agent) at (0,0) {RL Agent};
  \node[core] (env) at (5.4,0) {Environment};

  \draw[loop] (env.west) -- node[above] {$s_t$} (agent.east);
  \draw[loop] (agent.east) -- node[below] {$a_t$} (env.west);
  \draw[loop] (env.north west) .. controls +(0.8,1.1) and +(0.8,1.1) ..
      node[above] {$r_t,\;s_{t+1}$} (agent.north east);

  \node[module] (planner) at (0,2.7) {Planner};
  \node[module] (reward)  at (5.4,2.7) {Reward / Verifier};
  \node[module] (world)   at (5.4,-2.7) {World Model};
  \node[module] (tool)    at (0,-2.7) {Tool / Memory};

  \draw[mod] (planner.south) -- node[left] {\scriptsize subgoals} (agent.north);
  \draw[mod] (reward.south west) -- node[above] {\scriptsize reward signal} (agent.north east);
  \draw[mod] (world.north west) -- node[below] {\scriptsize imagined rollouts} (agent.south east);
  \draw[mod] (tool.north) -- node[left] {\scriptsize context / memory state} (agent.south);

  \draw[mod,dashed] (env.north) .. controls +(0.3,0.9) and +(0.3,-0.9) .. (reward.east);
  \draw[mod,dashed] (env.south) .. controls +(0.3,-0.9) and +(0.3,0.9) .. (world.east);
\end{tikzpicture}
\caption{LLM modules embedded in the RL loop as planner, reward/verifier, world model, and tool/memory interfaces that alter trajectory-level optimization and credit assignment.}
\label{fig:F1}
\end{figure*}
""".strip()


def tikz_figure_taxonomy() -> str:
    return r"""
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  font=\small,
  root/.style={rounded corners, draw=black, thick, align=center, fill=gray!10, minimum width=95mm, minimum height=10mm},
  leaf/.style={rounded corners, draw=black, thick, align=center, minimum width=36mm, minimum height=8.5mm},
  arr/.style={-Latex, thick}
]
  \node[root] (r) at (0,0) {Unified Taxonomy of LLM Roles in RL};

  \node[leaf] (a1) at (-6.1,-1.9) {Planner \&\\Decomposition};
  \node[leaf] (a2) at (-2.0,-1.9) {Policy \&\\Program Generation};
  \node[leaf] (a3) at (2.0,-1.9) {Reward Design\\\& Verifiers};
  \node[leaf] (a4) at (6.1,-1.9) {World Models\\\& Simulation};

  \node[leaf] (a5) at (-6.1,-4.3) {Exploration \&\\Curriculum};
  \node[leaf] (a6) at (-2.0,-4.3) {Offline Data\\Augmentation};
  \node[leaf] (a7) at (2.0,-4.3) {Tool \& Memory\\Interfaces};
  \node[leaf] (a8) at (6.1,-4.3) {Search\\Hybrids};

  \foreach \x in {a1,a2,a3,a4,a5,a6,a7,a8} {
    \draw[arr] (r.south) -- (\x.north);
  }
\end{tikzpicture}
\caption{Unified taxonomy of LLM roles in RL organized by optimization pathway.}
\label{fig:F2}
\end{figure*}
""".strip()


def pgfplots_year_category_heatmap(df: pd.DataFrame) -> str:
    years = sorted(int(y) for y in df["Year"].unique() if int(y) > 0)
    cats = [
        ("InfoProc", lambda ts: ("Reasoning" in ts) or ("LanguageConditioning" in ts)),
        ("Planner", lambda ts: "Planning" in ts),
        ("Policy", lambda ts: "Policy" in ts),
        ("Reward", lambda ts: ("Reward" in ts) or ("Verifier" in ts) or ("CreditAssignment" in ts)),
        ("World", lambda ts: ("WorldModel" in ts) or ("Simulation" in ts)),
        ("Explore", lambda ts: ("Exploration" in ts) or ("Curriculum" in ts)),
        ("Offline", lambda ts: ("OfflineRL" in ts) or ("DatasetGeneration" in ts)),
        ("ToolMem", lambda ts: ("ToolUse" in ts) or ("Memory" in ts)),
        ("Search", lambda ts: "TreeSearch" in ts),
    ]

    tag_sets = [tag_set(s) for s in df["Category_Tags"].tolist()]

    # Inline table for pgfplots (year, catIndex, value)
    rows = []
    for j, (_, pred) in enumerate(cats):
        for y in years:
            idxs = df.index[df["Year"] == y].tolist()
            v = 0
            for i in idxs:
                if pred(tag_sets[i]):
                    v += 1
            rows.append((y, j, v))

    data_lines = ["year cat value"] + [f"{y} {c} {v}" for (y, c, v) in rows]
    data_block = "\n".join(data_lines)

    mesh_cols = max(1, len(years))
    max_val = max((v for (_, _, v) in rows), default=0)

    ytick = ",".join(str(i) for i in range(len(cats)))
    yticklabels = ",".join([name for name, _ in cats])
    xtick = ",".join(str(y) for y in years)

    return rf"""
\begin{{figure*}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.82\textwidth,
  height=0.35\textwidth,
  xlabel={{Year}},
  ylabel={{Category}},
  xtick={{{xtick}}},
  ytick={{{ytick}}},
  yticklabels={{{yticklabels}}},
  x tick label style={{rotate=45, anchor=east}},
  colormap/viridis,
  colorbar,
  colorbar style={{ylabel={{Count}}}},
  point meta min=0,
  point meta max={max_val},
  enlargelimits=false,
  grid=both,
  major grid style={{draw=gray!20}},
  minor grid style={{draw=gray!10}},
]
\addplot[
  matrix plot*,
  mesh/cols={mesh_cols},
  point meta=explicit,
] table [meta=value, x=year, y=cat] {{
{data_block}
}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Year $\times$ category density across LLM-to-RL mechanism classes (counts).}}
\label{{fig:F3}}
\end{{figure*}}
""".strip()


def write_citation_report(
    all_keys: Set[str],
    cites_main: Set[str],
    cites_tables: Set[str],
    cites_appendix: Set[str],
    tex_path: Path,
) -> None:
    cited_any = cites_main | cites_tables | cites_appendix
    not_cited = sorted(all_keys - cited_any)
    appendix_only = sorted(cites_appendix - cites_main - cites_tables)
    tables_only = sorted(cites_tables - cites_main - cites_appendix)

    # Count total citation occurrences in the generated TeX.
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    cite_cmds = re.findall(r"\\cite\{([^}]+)\}", tex)
    total_cite_mentions = 0
    for chunk in cite_cmds:
        total_cite_mentions += len([k for k in chunk.split(",") if k.strip()])

    lines: List[str] = []
    lines.append("# Citation Coverage Report")
    lines.append("")
    lines.append(f"- Total verified papers in core_llm_for_rl_v3.csv: **{len(all_keys)}**")
    lines.append(f"- Unique papers cited in main text: **{len(cites_main)}**")
    lines.append(f"- Unique papers cited in tables (outside appendix): **{len(cites_tables)}**")
    lines.append(f"- Unique papers cited in appendix: **{len(cites_appendix)}**")
    lines.append(f"- Total citation mentions (all \\cite keys, including repeats): **{total_cite_mentions}**")
    lines.append("")
    lines.append("## Exclusivity Breakdown")
    lines.append("")
    lines.append(f"- Cited only in tables: **{len(tables_only)}**")
    lines.append(f"- Cited only in appendix: **{len(appendix_only)}**")
    lines.append(f"- Not cited anywhere: **{len(not_cited)}**")
    if not_cited:
        lines.append("")
        lines.append("## Not Cited (Should Be Empty)")
        lines.append("")
        for k in not_cited:
            lines.append(f"- {k}")
    CITATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(df: pd.DataFrame) -> None:
    # Canonical ID -> BibKey map (used for must-have classics and representative citations).
    keymap = {cid: bk for cid, bk in zip(df["CanonicalID"].tolist(), df["BibKey"].tolist())}

    def k(cid: str) -> str:
        cid2 = normalize_canonical_id(cid)
        if cid2 in keymap:
            return keymap[cid2]
        raise KeyError(cid2)

    # Landmark anchors.
    saycan = k("arXiv:2204.01691")
    cap = k("10.1109/icra48891.2023.10160591")
    voyager = k("arXiv:2305.16291")
    eureka = k("arXiv:2310.12931")
    text2reward = k("arXiv:2309.11489")

    all_keys = set(df["BibKey"].tolist())
    cites_main: Set[str] = set()
    cites_tables: Set[str] = set()
    cites_appendix: Set[str] = set()

    tags = df["Category_Tags"].astype(str)
    reps_plan = select_representatives_by_filter(df, tags.str.contains("Planning", regex=False), 8, must=[saycan, voyager])
    reps_policy = select_representatives_by_filter(df, tags.str.contains("Policy", regex=False), 8, must=[cap])
    reps_reward = select_representatives_by_filter(
        df,
        tags.str.contains("Reward", regex=False) | tags.str.contains("Verifier", regex=False) | tags.str.contains("CreditAssignment", regex=False),
        8,
        must=[eureka, text2reward],
    )
    reps_world = select_representatives_by_filter(df, tags.str.contains("WorldModel", regex=False) | tags.str.contains("Simulation", regex=False), 8)
    reps_offline = select_representatives_by_filter(df, tags.str.contains("OfflineRL", regex=False) | tags.str.contains("DatasetGeneration", regex=False), 8)
    reps_tool = select_representatives_by_filter(df, tags.str.contains("ToolUse", regex=False) | tags.str.contains("Memory", regex=False), 8, must=[voyager])
    reps_search = select_representatives_by_filter(df, tags.str.contains("TreeSearch", regex=False), 8)
    reps_explore = select_representatives_by_filter(
        df,
        tags.str.contains("Exploration", regex=False) | tags.str.contains("Curriculum", regex=False),
        8,
        must=[voyager],
    )
    reps_arch = select_representatives_by_filter(
        df,
        tags.str.contains("Architecture", regex=False) | tags.str.contains("HybridRL", regex=False),
        8,
    )

    # Appendix matrix (all papers; this is where paper-level details live).
    df_app = df.copy()
    df_app["PrimaryAxis"] = [primary_axis(tag_set(s)) for s in df_app["Category_Tags"].tolist()]
    df_app["RLSetting"] = [rl_setting_from_tags(tag_set(s)) for s in df_app["Category_Tags"].tolist()]
    df_app["Domain"] = [domain_from_summary(t, s) for t, s in zip(df_app["Title"], df_app["Contribution_Summary"])]

    app_rows: List[Tuple[str, str, str, str, str]] = []
    for _, r in df_app.sort_values(["Year", "Title"], ascending=[False, True]).iterrows():
        y = str(int(r["Year"])) if int(r["Year"]) else ""
        ax = latex_escape_text(abbrev_axis(r["PrimaryAxis"]))
        rs = latex_escape_text(abbrev_setting(r["RLSetting"]))
        dom = latex_escape_text(abbrev_domain(r["Domain"]))
        title = latex_escape_text(r["Title"])
        ck = r["BibKey"]
        paper_cell = title + " " + cite([ck], cites_appendix)
        app_rows.append((y, ax, rs, dom, paper_cell))

    appendix: List[str] = []
    appendix.append(r"\begin{appendices}")
    appendix.append(r"\section{Structured Paper Matrix}")
    appendix.append(
        "This appendix provides a structured matrix of the surveyed literature. "
        "Detailed paper-level information is placed here to preserve conceptual flow in the main text."
    )
    appendix.append("")
    appendix.append(r"\scriptsize")
    appendix.append(r"\setlength{\LTleft}{0pt}")
    appendix.append(r"\setlength{\LTright}{0pt}")
    appendix.append(r"\setlength{\tabcolsep}{2pt}")
    appendix.append(r"\begin{longtable}{@{}p{0.65cm}p{1.35cm}p{1.35cm}p{1.30cm}p{7.30cm}@{}}")
    appendix.append(r"\caption{Structured matrix of surveyed papers.}\label{tab:appendix_matrix}\\")
    appendix.append(r"\toprule")
    appendix.append(r"\textbf{Year} & \textbf{Role} & \textbf{RL Setting} & \textbf{Domain} & \textbf{Paper} \\")
    appendix.append(r"\midrule")
    appendix.append(r"\endfirsthead")
    appendix.append(r"\toprule")
    appendix.append(r"\textbf{Year} & \textbf{Role} & \textbf{RL Setting} & \textbf{Domain} & \textbf{Paper} \\")
    appendix.append(r"\midrule")
    appendix.append(r"\endhead")
    for row in app_rows:
        appendix.append(" & ".join(row) + r" \\")
    appendix.append(r"\bottomrule")
    appendix.append(r"\end{longtable}")
    appendix.append(r"\end{appendices}")
    appendix_block = "\n".join(appendix)

    # Main manuscript
    lines: List[str] = []
    lines.append("% Generated manuscript.")
    lines.append(r"\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}")
    lines.append("")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{amsmath,amssymb,amsfonts}")
    lines.append(r"\usepackage{amsthm}")
    lines.append(r"\usepackage[title]{appendix}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{tabularx}")
    lines.append(r"\usepackage{adjustbox}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{url}")
    lines.append(r"\usepackage{tikz}")
    lines.append(r"\usetikzlibrary{arrows.meta,positioning,calc}")
    lines.append("")
    lines.append(r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}")
    lines.append(r"\raggedbottom")
    lines.append("")
    lines.append(r"\makeatletter")
    lines.append(r"\@ifpackageloaded{hyperref}{%")
    lines.append(r"  \renewcommand{\theHtable}{\theHsection.\arabic{table}}%")
    lines.append(r"  \renewcommand{\theHfigure}{\theHsection.\arabic{figure}}%")
    lines.append(r"}{}")
    lines.append(r"\makeatother")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\title[LLM Modules in the RL Loop]{LLM Modules in the RL Loop: Roles, Architectures, and Credit Assignment Implications}")
    lines.append("")
    lines.append(r"\author*[1]{\fnm{Kourosh} \sur{Shahnazari}}")
    lines.append(r"\author[1]{\fnm{Seyed Moein} \sur{Ayyoubzadeh}}")
    lines.append(r"\affil[1]{\orgname{Sharif University of Technology}, \orgaddress{\city{Tehran}, \country{Iran}}}")
    lines.append("")

    abstract = (
        "Large language models are increasingly used as internal modules of reinforcement learning systems rather than external instruction interfaces. "
        "This survey develops a conceptual account of that shift: we analyze how planner, reward-verifier, world-model, search, data, and tool-memory modules alter trajectory optimization and credit assignment inside the RL loop. "
        "Instead of treating the literature as a catalog of isolated methods, we provide a unified framework that links module placement to supervision pathways, update bias, and failure propagation. "
        "The result is a theory-driven synthesis of architectural roles, optimization implications, and open research directions for reliable LLM-to-RL systems."
    )
    lines.append(r"\abstract{" + latex_escape_text(abstract) + r"}")
    lines.append(r"\keywords{large language models, reinforcement learning, planning, reward shaping, world models, credit assignment}")
    lines.append("")
    lines.append(r"\maketitle")
    lines.append(r"\noindent\textbf{Index Terms}---large language models, reinforcement learning, planner modules, reward verifiers, world models, credit assignment.")
    lines.append("")

    # 1 Introduction
    lines.append(r"\section{Introduction}")
    lines.append(r"\subsection{Motivation}")
    p = (
        "The core problem addressed in this survey is how language-model modules change reinforcement learning optimization when they are embedded inside the control loop. "
        "Classical RL struggles with sparse supervision, delayed consequences, and brittle exploration because scalar rewards alone rarely encode enough structure for long-horizon behavior. "
        "Moduleized LLM integration creates a structural shift: optimization is mediated by plans, verifier checks, imagined rollouts, and tool-conditioned context rather than by terminal rewards alone."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Scope and Positioning (LLM-to-RL vs RLHF)}")
    p = (
        "This survey focuses on LLM-to-RL systems in which pretrained language modules improve RL agents interacting with environments. "
        "This direction is distinct from RLHF-style pipelines whose primary objective is improving language-model output quality. "
        "The relevant unit of analysis here is trajectory optimization and credit assignment in environment-grounded control."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Contributions of This Survey}")
    p = (
        "The contributions are fourfold. First, we formalize LLM modules as intervention operators over policy updates, reward pathways, trajectory distributions, and state construction. "
        "Second, we introduce a unified taxonomy organized by optimization effects rather than by modality labels. "
        "Third, we analyze how module placement changes temporal credit assignment and where systematic bias enters learning. "
        "Fourth, we synthesize structural risks and research gaps for robust multi-module RL systems."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    p = (
        "Landmark systems such as grounded planning "
        + cite([saycan], cites_main)
        + ", programmatic policy execution "
        + cite([cap], cites_main)
        + ", open-ended skill acquisition "
        + cite([voyager], cites_main)
        + ", and language-generated rewards "
        + cite([eureka, text2reward], cites_main)
        + " illustrate the breadth of this paradigm."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Organization of the Paper}")
    p = (
        "Section 2 reviews RL and language-model foundations with emphasis on supervision pathways. "
        "Section 3 introduces a formal framework for module interventions and credit assignment. "
        "Section 4 presents the unified taxonomy. "
        "Sections 5 and 6 analyze evaluation practices and system-level failures. "
        "Section 7 outlines open problems, and Section 8 concludes."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    # 2 Background
    lines.append(r"\section{Background}")
    lines.append(r"\subsection{Reinforcement Learning and Credit Assignment}")
    p = (
        "RL optimizes policies through trajectory distributions induced by policy-environment interaction. "
        "Even when objective functions are compact, optimization quality depends on how trajectories are generated, filtered, and replayed. "
        "Long-horizon domains amplify this dependence because early decisions influence distant outcomes through deep causal chains."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Capabilities of Large Language Models}")
    p = (
        "Large language models contribute strong priors over procedure, decomposition, and interface semantics that are difficult to infer from sparse reward signals alone. "
        "When grounded by environment feedback, these priors can be converted into executable plans, policy sketches, reward code, or verifier checks. "
        "These capabilities are useful precisely because they operate at temporal and semantic scales that conventional RL updates do not represent explicitly."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Why LLM Modules Change RL Optimization Dynamics}")
    p = (
        "RL optimization is sensitive to how trajectories are generated and how supervision is routed. "
        "LLM modules alter both: they can change trajectory priors, shape reward pathways, and redefine state representation through tools and memory. "
        "As a result, performance changes are often better explained by intervention pathways than by policy architecture alone "
        + cite([saycan, cap, voyager], cites_main)
        + " "
        + cite([eureka, text2reward], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    # 3 Conceptual Framework
    lines.append(r"\section{Conceptual Framework: LLM Modules in the RL Loop}")
    lines.append(r"\subsection{Formal Definition}")
    p = (
        "We model each LLM component as an intervention operator $\mathcal{M}$ acting on the tuple $(s_t,a_t,r_t,\tau)$ that governs RL learning dynamics. "
        "Depending on role, the operator can transform state representation, generate trajectory priors, reshape supervision, or alter rollout composition before policy updates are applied."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Intervention Points in the RL Loop}")
    p = (
        "Let $\pi_\theta$ denote the trainable policy and $\tau \sim p_{\pi_\theta}(\tau)$ a trajectory. A generic policy-gradient update is $\nabla J(\theta)=\mathbb{E}_{\tau}[\sum_t \hat{A}_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)]$. "
        "LLM modules intervene by replacing $(s_t,a_t,r_t,\tau)$ with transformed variables $(\tilde{s}_t,\tilde{a}_t,\tilde{r}_t,\tilde{\tau})$, thereby changing both gradient direction and gradient variance."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Credit Assignment Implications}")
    p = (
        "Planner modules reshape trajectory priors before low-level execution, reward-verifier modules densify supervision pathways, world modules alter synthetic-real rollout balance, and tool-memory modules redefine what counts as state information. "
        "Each intervention can improve sample efficiency, but each can also inject structural bias if module authority is miscalibrated."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    p = (
        "Two recurrent error channels are planner-policy mismatch and verifier-induced reward hacking, where optimization shifts toward module artifacts rather than environment objectives "
        + cite(reps_plan[:4], cites_main)
        + " "
        + cite(reps_reward[:4], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Clean, Redesigned Figure 1}")
    lines.append(tikz_figure_loop())
    lines.append("")

    # 4 Unified taxonomy
    lines.append(r"\section{Taxonomy of LLM Roles in RL}")
    lines.append(tikz_figure_taxonomy())
    lines.append("")

    lines.append(r"\subsection{Planner \& Decomposition}")
    p = (
        "Planner-centric methods shape trajectory priors by converting abstract objectives into subgoals or option structures. "
        "Compared with flat policies, this reduces effective horizon and localizes credit across decomposition boundaries. "
        "The strongest systems keep planning adaptive under feedback rather than committing to static plans "
        + cite(reps_plan, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Policy \& Program Generation}")
    p = (
        "Policy-program approaches synthesize executable control templates that are grounded by RL feedback, execution traces, or critic signals. "
        "This interface can improve interpretability and recovery behavior, but it also introduces grounding risk when semantic plans and action-level feasibility diverge "
        + cite(reps_policy, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Reward Design \& Verifiers}")
    p = (
        "Reward-oriented modules densify supervision through language-derived shaping terms, code-level reward functions, and process verifiers. "
        "These mechanisms can dramatically improve learning under sparse feedback, but they may also induce proxy exploitation when checker objectives and task semantics diverge "
        + cite(reps_reward, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{World Models \& Simulation}")
    p = (
        "World-model modules inject imagined trajectories or surrogate dynamics into training. "
        "When calibrated, they improve sample efficiency by reallocating updates toward informative synthetic rollouts; when miscalibrated, they amplify model bias in value estimation "
        + cite(reps_world, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Exploration \& Curriculum}")
    p = (
        "Exploration and curriculum modules adapt task sequencing to competence and uncertainty, increasing coverage in sparse-reward regimes. "
        "The structural gain comes from matching exploration pressure to learning stage, while the structural risk is drift toward easy but low-value curricula "
        + cite(reps_explore, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Offline Data Augmentation}")
    p = (
        "Offline augmentation methods synthesize, relabel, or annotate trajectories to expand useful support for batch RL. "
        "Their effectiveness depends on conservative filtering and uncertainty-aware replay; otherwise synthetic traces can push updates outside realizable dynamics "
        + cite(reps_offline, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Tool \& Memory Interfaces}")
    p = (
        "Tool-memory interfaces treat retrieval outputs, API responses, and persistent traces as part of state construction. "
        "This improves long-horizon coherence in interactive tasks, yet introduces recursive failure channels when tool outputs are noisy or stale "
        + cite(reps_tool, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Search Hybrids}")
    p = (
        "Search-hybrid systems combine language priors with grounded branch evaluation and value correction. "
        "They improve deliberation under uncertainty by allocating compute to difficult decision points, but performance remains sensitive to branch-budget and pruning policy design "
        + cite(reps_search, cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    # 5 Benchmarks and evaluation ecosystem
    lines.append(r"\section{Benchmarks and Evaluation Ecosystem}")
    lines.append(r"\subsection{Domain Distribution}")
    p = (
        "Current evaluations are concentrated in embodied manipulation, web-interaction tasks, open-ended game-like environments, and continuous-control benchmarks. "
        "Each domain stresses different intervention pathways: robotics emphasizes grounded planning and safety, web tasks emphasize tool reliability, and open-ended settings emphasize memory and decomposition."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Benchmark Standardization Issues}")
    p = (
        "Cross-paper comparison remains difficult because benchmark suites vary widely in horizon length, compositionality, and tool constraints. "
        "Without standardized stress tests for module authority and intervention robustness, it is hard to separate genuine algorithmic progress from benchmark-specific convenience."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Metric Inconsistencies}")
    p = (
        "Return and success rate are necessary but not sufficient for moduleized systems. "
        "Reliable evaluation should include pathway-level diagnostics such as planner revision frequency, verifier disagreement, synthetic-rollout ratio, and tool-failure recovery behavior."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Trend Analysis Over Time}")
    p = (
        "A clear trend is the movement from single-module prototypes toward coupled architectures that combine planning, supervision restructuring, and interface mediation in one loop "
        + cite(reps_arch, cites_main)
        + ". "
        "This trend increases capability but also increases the need for theory-driven module authority control."
    )
    lines.append(p)
    lines.append("")

    # 6 Design trade-offs and failure modes
    lines.append(r"\section{Design Trade-offs and Failure Modes}")
    lines.append(r"\subsection{Planner--Policy Mismatch}")
    p = (
        "Planner--policy mismatch appears when semantically plausible subgoals are dynamically infeasible for the low-level controller. "
        "The effect is not merely execution failure; it distorts credit assignment because replay buffers accumulate trajectories that optimize decomposition artifacts rather than controllable progress "
        + cite(reps_plan[:5], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Reward Mis-specification}")
    p = (
        "Language-generated rewards can densify sparse supervision but can also overfit proxies that are easier to satisfy than task completion. "
        "When this occurs, optimization converges quickly to high proxy scores while true environment performance stagnates "
        + cite(reps_reward[:5], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Verifier Exploitation}")
    p = (
        "Verifier channels are vulnerable to exploitation when policy updates discover checker-specific shortcuts. "
        "Robust systems therefore combine verifiers with grounded rollouts, disagreement triggers, and conservative update gates that limit authority under uncertainty "
        + cite(reps_reward[2:7], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{World Model Hallucination}")
    p = (
        "Synthetic rollouts become harmful when model confidence exceeds model validity, producing biased value targets and brittle policy updates. "
        "Mitigation requires uncertainty-aware rollout weighting and explicit synthetic-data budgets so that imagination complements rather than overrides interaction evidence "
        + cite(reps_world[:5], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Tool-Mediated Error Propagation}")
    p = (
        "Tool outputs and memory traces can propagate transient errors across long trajectories, especially in multi-step interactive tasks. "
        "Typed interfaces, schema validation, and selective memory writes are therefore central to stability in tool-augmented loops "
        + cite(reps_tool[:5], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    lines.append(r"\subsection{Computational Overhead vs Performance}")
    p = (
        "Moduleized systems trade sample efficiency for inference and orchestration cost. "
        "Planning, verification, and search improve decision quality but increase latency and variance in wall-clock performance, requiring budget-aware scheduling policies "
        + cite(reps_search[:5] + reps_arch[:3], cites_main)
        + "."
    )
    lines.append(p)
    lines.append("")

    # 7 Open research problems
    lines.append(r"\section{Open Research Problems}")

    lines.append(r"\subsection{Formalizing module authority}")
    p = (
        "Current systems typically assign module influence through heuristic schedules or fixed weights. "
        "A formal treatment should model authority as a control variable with uncertainty-aware adaptation, allowing systems to reduce reliance on unstable modules under shift while preserving their benefits when confidence is high."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Calibrated verifier design}")
    p = (
        "Verifier performance is usually measured in-distribution, but deployment failures arise when tasks, tools, or environments change. "
        "Future verifier design must include calibration protocols, disagreement-based gating, and explicit uncertainty reporting so that process supervision remains reliable beyond benchmark regimes."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Uncertainty-aware world models}")
    p = (
        "Language-mediated world models need principled uncertainty propagation into policy updates. "
        "Without it, optimistic synthetic rollouts can induce systematic overestimation and brittle control; with it, synthetic data can be used selectively as a variance-reduction tool rather than an unbounded substitute for interaction."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Multi-module interaction theory}")
    p = (
        "Most analyses treat modules independently even though practical systems are coupled networks of planners, verifiers, world models, and interfaces. "
        "A useful theory should characterize interaction effects, including feedback amplification and failure cascades, to guide architecture-level regularization."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Safety and alignment in RL loops}")
    p = (
        "Safety in LLM-to-RL systems is an optimization problem over module interactions, not an afterthought layered onto final policy outputs. "
        "Research should define fail-safe intervention policies, rollback mechanisms, and cross-module consistency checks that remain effective under long-horizon uncertainty."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    lines.append(r"\subsection{Scaling laws for LLM-to-RL}")
    p = (
        "The field lacks scaling laws that connect model size, module placement, interaction budget, and performance gain. "
        "Establishing such laws would allow principled allocation of compute across planning, verification, and simulation channels instead of ad-hoc architecture growth."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    # 9 Conclusion
    lines.append(r"\section{Conclusion}")
    p = (
        "LLM-to-RL is best understood as a structural reconfiguration of reinforcement learning optimization. "
        "By embedding language modules into planning, supervision, world construction, and interface mediation, modern systems change how trajectories are generated, interpreted, and credited. "
        "The next phase of progress will depend on principled authority control, calibrated supervision, and evaluation protocols that expose module-level causality rather than only aggregate outcomes."
    )
    lines.append(latex_escape_text(p))
    lines.append("")

    # Appendix and bibliography
    lines.append(appendix_block)
    lines.append("")
    lines.append(r"\bibliography{sn-bibliography}")
    lines.append(r"\end{document}")

    TEX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Coverage report and strict coverage gate.
    write_citation_report(all_keys, cites_main, cites_tables, cites_appendix, TEX_OUT)
    missing = sorted(all_keys - (cites_main | cites_tables | cites_appendix))
    if missing:
        raise SystemExit(f"Coverage failure: {len(missing)} papers not cited. Example: {missing[:5]}")

def main() -> None:
    if not CORE_CSV.exists():
        raise SystemExit("Missing core_llm_for_rl_v3.csv at repo root.")

    ensure_bst_present()

    df = pd.read_csv(CORE_CSV).fillna("")
    for col in ["Title", "Authors", "Venue", "ArXiv_or_DOI", "Category_Tags", "Contribution_Summary"]:
        df[col] = df[col].apply(clean_space)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)

    df["CanonicalID"] = df["ArXiv_or_DOI"].apply(normalize_canonical_id)
    df["BibKey"] = df["CanonicalID"].apply(canonical_id_to_bibkey)

    # Defensive BibKey de-duplication.
    seen: Dict[str, int] = {}
    fixed: List[str] = []
    for k0 in df["BibKey"].tolist():
        if k0 not in seen:
            seen[k0] = 1
            fixed.append(k0)
        else:
            seen[k0] += 1
            fixed.append(f"{k0}_{seen[k0]}")
    df["BibKey"] = fixed

    # Remove old matplotlib-exported figures if present (figures are now TikZ/PGFPlots in LaTeX).
    fig_dir = PAPER_DIR / "figures"
    if fig_dir.exists():
        for name in ["F1_llm_rl_loop.pdf", "F2_taxonomy_tree.pdf", "F3_year_category_heatmap.pdf"]:
            p = fig_dir / name
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    write_bib(df)
    write_tex(df)

    print(f"Generated: {TEX_OUT}, {BIB_OUT}, {CITATION_REPORT}")


if __name__ == "__main__":
    main()
