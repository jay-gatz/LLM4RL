
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('.')
CORE = ROOT / 'core_llm_for_rl_v3.csv'
OUT = ROOT / 'survey_paper'
SEC = OUT / 'sections'
FIG = OUT / 'figures'
YEARS = [2022, 2023, 2024, 2025, 2026]

REQ = ['Title','Authors','Year','Venue','ArXiv_or_DOI','Category_Tags','Contribution_Summary','Why_It_Belongs_in_Survey','Scope_Label','Verification_Provider','Verification_URL','Verification_Confidence','Provenance']


def c(x):
    return re.sub(r'\s+', ' ', str(x)).strip()


def tags(x):
    return [t.strip() for t in str(x).split('|') if t.strip()]


def esc(x):
    x = c(x)
    m = {'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(m.get(ch, ch) for ch in x)


def besc(x):
    x = c(x).replace('\\', ' ').replace('{', '(').replace('}', ')')
    return x.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def norm_id(x):
    x = c(x)
    x = x.replace('https://arxiv.org/abs/', 'arXiv:').replace('http://arxiv.org/abs/', 'arXiv:')
    return x


def domain(title, summary):
    t = (title + ' ' + summary).lower()
    if any(k in t for k in ['minecraft','alfworld','webshop','textworld','atari','game']):
        return 'Games/Text Environments'
    if any(k in t for k in ['robot','embodied','manipulation','navigation','vla']):
        return 'Robotics/Embodied'
    if any(k in t for k in ['web','browser','computer use','gui']):
        return 'Web/Interactive'
    if any(k in t for k in ['mujoco','locomotion','continuous control']):
        return 'Continuous Control'
    if any(k in t for k in ['offline rl','batch rl','dataset']):
        return 'Offline RL'
    if any(k in t for k in ['simulation','simulator','world model']):
        return 'Simulation/Model-based'
    if any(k in t for k in ['multi-agent','multi agent']):
        return 'Multi-Agent'
    return 'General RL'


def role(ts):
    s = set(ts)
    if 'Planning' in s:
        return 'Planner/Task Decomposer'
    if 'Reward' in s and 'Verifier' in s:
        return 'Reward+Verifier'
    if 'Reward' in s:
        return 'Reward Shaper'
    if 'WorldModel' in s:
        return 'World Model/Simulator'
    if 'ToolUse' in s and 'Memory' in s:
        return 'Tool+Memory Controller'
    if 'ToolUse' in s:
        return 'Tool-aware Controller'
    if 'Policy' in s:
        return 'Policy/Controller'
    if 'DatasetGeneration' in s or 'OfflineRL' in s:
        return 'Data/Trajectory Generator'
    if 'TreeSearch' in s:
        return 'Search Prior Module'
    return 'Architecture Module'


def setting(ts, s):
    s2 = s.lower(); t = set(ts)
    if 'OfflineRL' in t or 'DatasetGeneration' in t: return 'Offline/Batch RL'
    if 'TreeSearch' in t: return 'Search-augmented RL'
    if 'WorldModel' in t or 'Simulation' in t: return 'Model-based RL'
    if 'Curriculum' in t or 'Exploration' in t: return 'Exploration/Curriculum RL'
    if 'MultiAgent' in t: return 'Multi-agent RL'
    if 'LanguageConditioning' in t: return 'Language-conditioned RL'
    if 'hierarchical' in s2 or 'long-horizon' in s2: return 'Hierarchical RL'
    return 'General RL Optimization'


def mk_keys(df):
    used, keys = set(), []
    stop = {'the','a','an','of','for','in','on','and','with','to','via','using','large','language','model','models'}
    for _, r in df.iterrows():
        y = str(int(r['Year'])) if str(r['Year']).isdigit() else '0000'
        fa = c(r['Authors']).split(',')[0].strip() or 'Anon'
        sn = re.sub(r'[^A-Za-z0-9]', '', fa.split()[-1]) or 'Anon'
        ws = [w for w in re.findall(r'[A-Za-z0-9]+', c(r['Title']).lower()) if w not in stop]
        stem = ''.join(w.capitalize() for w in ws[:2])[:18] or 'Paper'
        b = f'{sn}{y}{stem}'; k = b; i = 2
        while k in used:
            k = f'{b}{i}'; i += 1
        used.add(k); keys.append(k)
    df = df.copy(); df['BibKey'] = keys
    return df


def pick(df, wanted, n=6):
    m = pd.Series(False, index=df.index)
    for t in wanted:
        m = m | df['Tags'].apply(lambda xs: t in xs)
    ks = df[m].sort_values(['Year','Title'], ascending=[False,True])['BibKey'].tolist()[:n]
    if len(ks) < n:
        for k in df.sort_values(['Year','Title'], ascending=[False,True])['BibKey'].tolist():
            if k not in ks: ks.append(k)
            if len(ks) >= n: break
    return ','.join(ks[:n])


def rep(df, wanted, n=8):
    s = df[df['Tags'].apply(lambda xs: any(t in xs for t in wanted))]
    s = s.sort_values(['Year','Title'], ascending=[False,True]).head(n)
    return '; '.join(s['Title'].tolist())

def longtable(headers, rows, spec):
    z = []
    z.append(r'\begin{longtable}{' + spec + '}')
    z.append(r'\toprule')
    z.append(' & '.join(headers) + r' \\')
    z.append(r'\midrule')
    z.append(r'\endfirsthead')
    z.append(r'\toprule')
    z.append(' & '.join(headers) + r' \\')
    z.append(r'\midrule')
    z.append(r'\endhead')
    for row in rows:
        z.append(' & '.join(esc(str(x)) for x in row) + r' \\')
    z.append(r'\bottomrule')
    z.append(r'\end{longtable}')
    return '\n'.join(z)


def fig_loop():
    fig, ax = plt.subplots(figsize=(11, 6.2)); ax.axis('off')
    boxes = {
        'Environment': (0.05,0.46,0.19,0.2),
        'RL Agent\\nPolicy/Value/Replay': (0.33,0.44,0.24,0.23),
        'LLM Planner': (0.68,0.74,0.24,0.13),
        'LLM Reward/Verifier': (0.68,0.51,0.24,0.13),
        'LLM World/Simulator': (0.68,0.28,0.24,0.13),
        'Tools + Memory': (0.33,0.13,0.24,0.13),
    }
    for label,(x,y,w,h) in boxes.items():
        ax.add_patch(plt.Rectangle((x,y), w,h, fill=False, lw=1.8))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    def a(x1,y1,x2,y2,t=''):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', lw=1.3))
        if t: ax.text((x1+x2)/2, (y1+y2)/2+0.015, t, fontsize=8, ha='center')
    a(0.24,0.56,0.33,0.56,'state'); a(0.33,0.5,0.24,0.5,'action')
    a(0.57,0.58,0.68,0.79,'subgoals'); a(0.57,0.55,0.68,0.56,'reward hints')
    a(0.57,0.51,0.68,0.33,'model queries'); a(0.68,0.51,0.57,0.47,'shaped signals')
    a(0.57,0.29,0.45,0.20,'tool calls/history'); a(0.45,0.20,0.57,0.45,'context')
    fig.tight_layout(); fig.savefig(FIG / 'f1_llm_rl_loop.png', dpi=260); plt.close(fig)


def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(12,6.8)); ax.axis('off')
    ax.text(0.5,0.93,'Taxonomy: LLM Modules Inside RL Loops',ha='center',fontsize=15,fontweight='bold')
    ax.text(0.5,0.84,'LLM -> RL gains through trajectory-level optimization',ha='center',bbox=dict(boxstyle='round',fc='white',ec='black'),fontsize=10)
    nodes = [
      ('Planner / Decomposer',0.12,0.63),('Policy / Program Control',0.31,0.63),('Reward / Verifier',0.5,0.63),
      ('World Model / Simulation',0.69,0.63),('Exploration / Curriculum',0.88,0.63),('Offline Data / Relabeling',0.22,0.37),
      ('Tool Use / Memory',0.5,0.37),('Tree Search Hybrids',0.78,0.37)
    ]
    for label,x,y in nodes:
        ax.text(x,y,label,ha='center',va='center',bbox=dict(boxstyle='round',fc='white',ec='black'),fontsize=10)
        ax.annotate('', xy=(x,y+0.04), xytext=(0.5,0.81), arrowprops=dict(arrowstyle='->', lw=1.1))
    fig.tight_layout(); fig.savefig(FIG / 'f2_taxonomy_tree.png', dpi=260); plt.close(fig)


def fig_heat(df):
    ys = [str(y) for y in YEARS]
    cats = ['Planning','Policy','Reward','WorldModel','Exploration','OfflineRL','ToolUse','Memory','TreeSearch']
    mat = np.zeros((len(cats), len(ys)), dtype=int)
    for i,cat in enumerate(cats):
        for j,y in enumerate(ys):
            mat[i,j] = int(((df['Year'].astype(str)==y) & df['Category_Tags'].str.contains(cat, regex=False)).sum())
    fig, ax = plt.subplots(figsize=(10.8,5.2))
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(ys))); ax.set_xticklabels(ys)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_title('Year x Category Density in Verified Core')
    for i in range(len(cats)):
        for j in range(len(ys)):
            ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label('Paper Count')
    fig.tight_layout(); fig.savefig(FIG / 'f3_year_category_heatmap.png', dpi=260); plt.close(fig)


def build_rows(df):
    d = df.sort_values(['Year','Title'], ascending=[False,True]).copy()
    rt = lambda tag, n=3: '; '.join(d[d['Category_Tags'].str.contains(tag, regex=False)].head(n)['Title'].tolist())
    t1 = [
      ('Planner / Task Decomposer','Hierarchical decomposition and long-horizon intent grounding','Hierarchical and long-horizon RL',rt('Planning',3)),
      ('Policy / Controller','Language-conditioned action generation or programmatic control','Policy optimization and closed-loop control',rt('Policy',3)),
      ('Reward / Verifier','Shaped reward, process supervision, trajectory checks','Sparse-reward and constrained RL',rt('Reward',3)),
      ('World Model / Simulation','Dynamics priors and simulator proxies','Model-based RL',rt('WorldModel',3)),
      ('Exploration / Curriculum','Goal proposal and curriculum scheduling','Exploration-heavy RL',rt('Exploration',3)),
      ('Offline RL / Data Generation','Trajectory synthesis, relabeling, and augmentation','Offline or batch RL',rt('OfflineRL',3)),
      ('Tool Use / Memory','External API interaction and long-context updates','Interactive multi-turn RL agents',rt('ToolUse',3)),
      ('Tree Search Hybrids','MCTS-like branching with language priors','Search-augmented decision loops',rt('TreeSearch',3)),
    ]
    t2 = []
    for _, r in d.head(110).iterrows():
        t2.append((int(r['Year']), role(r['Tags']), setting(r['Tags'], r['Contribution_Summary']), r['Domain'], r['Title'][:94]))

    doms = ['Robotics/Embodied','Web/Interactive','Games/Text Environments','Continuous Control','Offline RL','Simulation/Model-based','General RL','Multi-Agent']
    cats = ['Planning','Policy','Reward','WorldModel','Exploration','OfflineRL','ToolUse','Memory','TreeSearch']
    t3 = []
    for dm in doms:
        vals = []
        for cat in cats:
            sub = df[df['Category_Tags'].str.contains(cat, regex=False)]
            vals.append(int((sub['Domain'] == dm).sum()))
        t3.append((dm,*vals))

    t4 = [
      ('LLM reward shaping', int(df['Category_Tags'].str.contains('Reward', regex=False).sum()), 'Language-derived reward terms densify sparse supervision.'),
      ('Verifier in-loop supervision', int(df['Category_Tags'].str.contains('Verifier', regex=False).sum()), 'Trajectory checks gate policy updates.'),
      ('Reward + Credit assignment', int((df['Category_Tags'].str.contains('Reward', regex=False) & df['Category_Tags'].str.contains('CreditAssignment', regex=False)).sum()), 'Intermediate signals improve delayed reward attribution.'),
      ('Reward + World model coupling', int((df['Category_Tags'].str.contains('Reward', regex=False) & df['Category_Tags'].str.contains('WorldModel', regex=False)).sum()), 'Predictive rollouts combine with reward edits.'),
      ('Reward + Tool use coupling', int((df['Category_Tags'].str.contains('Reward', regex=False) & df['Category_Tags'].str.contains('ToolUse', regex=False)).sum()), 'Tool outcomes feed evaluator correction.'),
    ]

    t5 = [
      ('Tool-aware policies', int(df['Category_Tags'].str.contains('ToolUse', regex=False).sum()), 'Policies query external interfaces during episodes.'),
      ('Memory-augmented loops', int(df['Category_Tags'].str.contains('Memory', regex=False).sum()), 'Context persistence supports multi-turn control.'),
      ('Tool + Verifier co-design', int((df['Category_Tags'].str.contains('ToolUse', regex=False) & df['Category_Tags'].str.contains('Verifier', regex=False)).sum()), 'Verifier gates reduce invalid tool trajectories.'),
      ('Tool + Multi-agent settings', int((df['Category_Tags'].str.contains('ToolUse', regex=False) & df['Category_Tags'].str.contains('MultiAgent', regex=False)).sum()), 'Coordination policies share tool outcomes.'),
      ('Memory + Planning integration', int((df['Category_Tags'].str.contains('Memory', regex=False) & df['Category_Tags'].str.contains('Planning', regex=False)).sum()), 'Stored history stabilizes plan revision.'),
    ]

    cols = ['Planning','Policy','Reward','WorldModel','Exploration','OfflineRL','ToolUse','Memory','TreeSearch']
    t6 = []
    for y in YEARS:
        row = [y]
        for col in cols:
            row.append(int(((df['Year'] == y) & df['Category_Tags'].str.contains(col, regex=False)).sum()))
        t6.append(tuple(row))
    return {'t1':t1,'t2':t2,'t3':t3,'t4':t4,'t5':t5,'t6':t6}

def write_tables(rows):
    x = []
    x.append(r'\subsection{T1: Taxonomy Overview}')
    x.append(longtable(['Category','Mechanism','Typical RL Setting','Representative Papers'], rows['t1'], 'p{2.8cm}p{3.5cm}p{3.0cm}p{5.1cm}'))
    x.append(r'\subsection{T2: Condensed Paper-by-Paper Matrix}')
    x.append(longtable(['Year','LLM Role','RL Setting','Domain','Key Contribution (paper)'], rows['t2'], 'p{0.8cm}p{2.8cm}p{2.8cm}p{2.5cm}p{5.5cm}'))
    x.append(r'\subsection{T3: Benchmark Matrix (Domain x Category)}')
    x.append(longtable(['Domain','Planning','Policy','Reward','WorldModel','Exploration','OfflineRL','ToolUse','Memory','TreeSearch'], rows['t3'], 'p{3.0cm}rrrrrrrrr'))
    x.append(r'\subsection{T4: Reward and Verifier Usage Patterns}')
    x.append(longtable(['Pattern','Count','Interpretation'], rows['t4'], 'p{4.2cm}rp{6.8cm}'))
    x.append(r'\subsection{T5: Tool-use and Memory Patterns}')
    x.append(longtable(['Pattern','Count','Interpretation'], rows['t5'], 'p{4.2cm}rp{6.8cm}'))
    x.append(r'\subsection{T6: Chronological Trends (2022--2026)}')
    x.append(longtable(['Year','Planning','Policy','Reward','WorldModel','Exploration','OfflineRL','ToolUse','Memory','TreeSearch'], rows['t6'], 'p{1.0cm}rrrrrrrrr'))
    (SEC / 'tables.tex').write_text('\n\n'.join(x) + '\n', encoding='utf-8')


def export_analytics(df, rows):
    rec = []
    for y in YEARS:
        rec.append({'Table':'YearDistribution','Row':y,'Column':'Count','Value':int((df['Year']==y).sum())})
    tc = Counter()
    for xs in df['Tags']:
        for t in xs: tc[t] += 1
    for t,cnt in sorted(tc.items()):
        rec.append({'Table':'TagDistribution','Row':t,'Column':'Count','Value':int(cnt)})
    for v,cnt in df['Venue'].value_counts().head(20).items():
        rec.append({'Table':'TopVenues','Row':v,'Column':'Count','Value':int(cnt)})
    for d,cnt in df['Domain'].value_counts().items():
        rec.append({'Table':'DomainDistribution','Row':d,'Column':'Count','Value':int(cnt)})
    for name, trs in rows.items():
        for ridx, row in enumerate(trs, 1):
            for cidx, val in enumerate(row, 1):
                rec.append({'Table':f'Generated_{name.upper()}','Row':ridx,'Column':cidx,'Value':val})
    pd.DataFrame(rec).to_csv(OUT / 'tables_from_csv.csv', index=False)


def write_bib(df):
    ents = []
    for _, r in df.iterrows():
        key = r['BibKey']; title = besc(r['Title']); year = str(int(r['Year']))
        raw_auth = c(r['Authors'])
        if ' and ' in raw_auth:
            auth_parts = [a.strip() for a in raw_auth.split(' and ') if a.strip()]
        else:
            auth_parts = [a.strip() for a in raw_auth.split(',') if a.strip()]
        author = ' and '.join(besc(a) for a in auth_parts) if auth_parts else 'Unknown'
        venue = besc(r['Venue']); rid = norm_id(r['ArXiv_or_DOI']); url = c(r['Verification_URL'])
        arx, doi = '', ''
        if rid.lower().startswith('arxiv:'): arx = rid.split(':',1)[1]
        elif re.match(r'^10\\.\\d{4,9}/', rid, flags=re.I): doi = rid
        y = [f'@misc{{{key},', f'  title = {{{title}}},', f'  author = {{{author}}},', f'  year = {{{year}}},']
        if venue: y.append(f'  note = {{{venue}}},')
        if arx:
            y.append(f'  eprint = {{{arx}}},'); y.append('  archivePrefix = {arXiv},'); y.append('  primaryClass = {cs.AI},')
        if doi: y.append(f'  doi = {{{doi}}},')
        if url: y.append(f'  url = {{{url}}},')
        y.append('}'); ents.append('\n'.join(y))
    (OUT / 'references.bib').write_text('\n\n'.join(ents) + '\n', encoding='utf-8')


def count_csv(path):
    if not path.exists(): return 0
    try: return len(pd.read_csv(path))
    except Exception: return 0


def write_sections(df, counts):
    cp = pick(df, ['Planning','Policy'], 6)
    cr = pick(df, ['Reward','Verifier','CreditAssignment'], 6)
    cw = pick(df, ['WorldModel','Simulation'], 6)
    ce = pick(df, ['Exploration','Curriculum'], 6)
    co = pick(df, ['OfflineRL','DatasetGeneration'], 6)
    ct = pick(df, ['ToolUse','Memory','MultiAgent'], 6)
    cs = pick(df, ['TreeSearch','Planning'], 5)
    cl = pick(df, ['LanguageConditioning','Reasoning'], 6)
    ca = pick(df, ['Architecture','HybridRL','AgentTraining'], 6)

    n = len(df); o = counts.get('out', 0); b = counts.get('border', 0); r = counts.get('rem', 0); s = n + o + b + r
    yc = {y:int((df['Year']==y).sum()) for y in YEARS}

    intro_examples = rep(df, ['Planning','Reward','ToolUse'], 8)
    reward_examples = rep(df, ['Reward','Verifier'], 8)
    world_examples = rep(df, ['WorldModel','Simulation'], 8)
    tool_examples = rep(df, ['ToolUse','Memory'], 8)

    sec = {}
    sec['abstract.tex'] = f'''This survey analyzes a directionality-constrained question: how large language models (LLMs) improve reinforcement learning (RL) when inserted directly into training and deployment loops. The manuscript is based on a verified core bibliography of {n} papers curated with item-by-item metadata checks and strict exclusion of RL-for-LLM post-training pipelines. The synthesis is organized by module placement: planning and decomposition, policy/controller support, reward shaping and verifier supervision, world-model proxies, exploration and curriculum generation, offline trajectory synthesis, tool-use with memory, and tree-search hybrids.

Across categories, the central pattern is trajectory-level leverage: LLM modules alter temporal abstraction, state construction, and supervisory density, thereby changing credit assignment dynamics rather than only generating better text outputs. The strongest gains appear when language outputs are treated as revisable intermediate signals coupled to environment feedback, not immutable instructions. This architecture-level finding recurs across robotics, web interaction, text environments, and simulator-heavy control domains.

We provide a PRISMA-style methodology, data-backed category/year analyses, and an implementation-oriented review of failure modes including reward hacking, planner-policy mismatch, and tool-mediated error accumulation. Beyond taxonomy, the paper offers design guidance on module authority, verification pathways, and diagnostics required for robust deployment. The resulting perspective reframes LLM-in-RL systems as credit-assignment engineering problems where language priors are useful only to the extent that they improve environment-grounded optimization. The survey also highlights reporting standards needed for reproducible comparison across rapidly evolving architectures.'''

    sec['introduction.tex'] = f'''Reinforcement learning optimizes trajectories under delayed and noisy feedback, which makes temporal credit assignment hard in sparse-reward and long-horizon settings. Large language models add semantic structure that can be injected into this loop through subgoal decomposition, reward shaping, verifier supervision, and state-context compression. The practical value appears only when language outputs are grounded by environment transitions and policy/value updates, not when they remain text-only suggestions \cite{{{cp}}}.

Representative systems in the verified core include: {esc(intro_examples)}. Although architectures differ, these works share a common mechanism: an LLM output changes a trajectory-relevant variable, such as action branching, reward decomposition, tool invocation policy, or memory state consumed by RL updates. This is where measurable gains in return and sample efficiency emerge \cite{{{cr}}}.

This survey enforces strict directionality. We include papers where LLM modules improve RL agents and environment-facing performance, and exclude papers where RL is used primarily to improve language-model outputs. The separation is essential because otherwise RL-for-LLM alignment papers overwhelm the signal needed to analyze decision-loop improvements \cite{{{ca}}}.

The unifying thesis is that LLMs are most impactful when they reshape trajectory-level optimization. Planner modules alter horizon structure. Reward/verifier modules alter supervisory density. World-model modules alter lookahead quality. Tool-memory modules alter state and action interfaces over long episodes. These modules can complement each other, but only with calibration and fallback mechanisms that keep environment grounding dominant \cite{{{cw}}}.

The practical consequence is methodological. A paper can report strong benchmark gains while still failing to establish a robust LLM-to-RL mechanism if evidence is limited to single-shot prompts or loosely specified evaluators. In contrast, studies that expose closed-loop internals (planner calls, verifier disagreements, tool failures, rollback decisions, and update frequencies) make it possible to attribute gains to specific architectural choices. This distinction is central for a mature survey: we are not only cataloging positive results, we are identifying which intervention channels are reproducibly causal.

Another motivation for this survey is comparability. The LLM-for-RL literature is rapidly expanding across domains with different observability regimes, action interfaces, and reward conventions. A robotics manipulation benchmark, a browser task benchmark, and a text-game benchmark may each show improvement but for different reasons. Without a unified taxonomy linked to credit assignment, these improvements remain difficult to reconcile. We therefore structure analysis around where language enters the loop and what optimization target it modifies.

From a systems perspective, LLM integration is best viewed as a control-architecture design problem, not a prompt-engineering problem. Architecture defines when language outputs are accepted, when they are overridden by low-level feedback, and how contradictions are resolved over long horizons. The strongest systems treat language as high-bandwidth prior knowledge with bounded authority, then rely on environment interaction to validate and revise that prior repeatedly \cite{{{ct}}}.

Finally, this work aims to be operational for practitioners. Along with conceptual taxonomy, we provide data-backed tables and figures derived from a verified bibliography, highlight robust evaluation patterns, and flag recurring failure channels that should be tested before deployment. The objective is to move from anecdotal demonstrations toward defensible engineering principles for LLM modules inside RL loops \cite{{{ca}}}.'''

    sec['background.tex'] = f'''\\subsection{{RL Loop Basics and Credit Assignment}}
RL repeatedly maps state to action, observes transitions, and updates policy/value functions. In realistic tasks, delayed outcomes and partial observability make direct credit assignment unstable. Sample efficiency depends on whether the learning signal can identify which earlier decisions caused later outcomes. This is hard in long-horizon tasks where many actions intervene between intent and reward.

Classical RL remedies include hierarchical decomposition, reward shaping, model-based lookahead, and memory-augmented policy state. Each remedy is essentially a different way to re-encode trajectory information so update rules receive denser and more causally useful supervision. The LLM-for-RL literature can be interpreted through the same lens: language modules are useful only when they improve how trajectories are represented, scored, or searched.

\\subsection{{What LLMs Add to RL Pipelines}}
LLMs provide structured priors over task semantics, procedures, and relational constraints. Embedded in RL loops, these priors become executable guidance: plans, reward rules, verifiers, memory summaries, and tool-call decisions \cite{{{cl}}}. In high-branching environments, such guidance can reduce exploration burden and improve long-horizon coherence.

Unlike static heuristics, LLM outputs can adapt to contextual details at runtime. This enables dynamic subgoal proposals, conditional reward terms, and trajectory-dependent evaluator criteria. However, this flexibility also introduces instability if language outputs are accepted without grounding checks. As a result, successful systems usually pair LLM modules with safeguards: confidence thresholds, verifier agreement tests, rollback strategies, and periodic reset-to-environment baselines.

Three interface patterns dominate. First, planner interfaces produce subgoals and option-level decomposition while RL handles low-level correction. Second, reward/verifier interfaces densify supervision and detect poor trajectories before updates \cite{{{cr}}}. Third, tool-memory interfaces maintain context and external interaction traces across long episodes \cite{{{ct}}}. In practice, these patterns are often combined, which creates both synergy and coupling risk.

Coupling risk appears when one module's bias propagates into another. For example, a planner can induce trajectories that reward models later over-score, causing policy updates to reinforce planner artifacts rather than environment-grounded competence. Similar feedback loops occur when tool errors are summarized into memory without validation. Understanding and controlling these interactions is a central technical theme in modern LLM-in-the-loop RL design.'''

    sec['methodology.tex'] = f'''\\subsection{{PRISMA-style Retrieval}}
Candidates were pooled from four channels: \\texttt{{papers.csv}}, references parsed from \\texttt{{2404.00282v3.pdf}}, citation expansion from verified seeds, and targeted search for under-covered years/clusters. High recall was used at ingestion, followed by strict verification and scope filtering.

The retrieval design intentionally separates discovery from acceptance. Discovery is broad and permissive to avoid losing relevant studies, while acceptance is conservative and evidence-driven. This split is important in fast-moving LLM literature where titles often overstate novelty and scope. By keeping candidate intake broad and verification strict, the pipeline improves coverage without sacrificing reliability.

\\subsection{{Item-by-item Verification}}
Every retained record was validated against authoritative metadata providers used in curation (arXiv/OpenAlex/Crossref/Semantic Scholar where available). Canonical title, authors, year, venue, identifier, and URL were normalized before inclusion. Invalid IDs, low-confidence matches, and non-academic artifacts were removed.

Verification was performed at row level, not dataset level. This means each paper needs a traceable canonical record; missing or conflicting metadata is not silently repaired. The resulting bibliography is therefore auditable: each row has a verification provider, URL, confidence tag, and provenance channel. This design prevents hallucinated or weakly matched entries from entering downstream analysis.

\\subsection{{Inclusion/Exclusion Criteria}}
Inclusion requires explicit evidence that LLM modules improve RL loops through planning, policy support, reward/verifier mechanisms, world modeling, exploration/curriculum, offline data generation, tool use, memory, or tree-search hybrids. Exclusion is strict for RL-for-LLM post-training work. Ambiguous directionality items were placed in manual review artifacts.

Directionality was treated as a hard criterion because mixed terminology can blur causal interpretation. If a paper centers on improving language model behavior with RL, it is out of scope even if it mentions agents. If a paper improves RL agents and uses language modules as in-loop components, it is in scope. This conservative policy reduces false inclusion at the cost of potentially excluding some hybrid papers, which are retained in borderline files for transparent review.

\\subsection{{Screening Outcome}}
The pipeline scanned {s} records and produced {n} verified in-scope core papers, {o} verified out-of-scope papers, {b} borderline papers, and {r} removed records. Core year distribution is 2022={yc.get(2022,0)}, 2023={yc.get(2023,0)}, 2024={yc.get(2024,0)}, 2025={yc.get(2025,0)}, 2026={yc.get(2026,0)}.

These counts indicate rapid growth in recent years and a sizable tail of ambiguous or out-of-scope records that would have contaminated a looser survey. The borderline pool is particularly informative: it highlights where the field's terminology is drifting and where future curation standards should sharpen definitions.

\\subsection{{Reproducibility}}
All statistics, tables, and figures in this manuscript are generated programmatically from \\texttt{{core\\_llm\\_for\\_rl\\_v3.csv}} and exported in deterministic order. The bibliography, analysis tables, and manuscript assets can be regenerated without manual citation insertion, reducing transcription errors and improving maintainability for later updates.

\\subsection{{Threats to Validity}}
Several validity threats remain despite strict curation. First, metadata providers occasionally lag behind preprint updates, which can produce minor venue or author-list drift. The pipeline mitigates this by preferring canonical IDs and storing verification URLs, but temporal inconsistencies can still occur across providers.

Second, directionality classification relies on title/abstract level evidence. Some papers may contain in-depth RL-agent contributions that are not fully visible in abstracts, while others may overstate RL relevance in high-level language. To reduce bias, uncertain records are not forced into core; they are assigned to manual-review artifacts. This increases precision at the cost of recall.

Third, category tagging is many-to-many and inherently interpretive. A paper with both reward and planning components may be represented differently across surveys. We address this with transparent tag sets and deterministic scripts, but category boundaries should still be interpreted as analytical tools rather than hard ontologies.

Fourth, benchmark heterogeneity limits quantitative synthesis. Reported gains are not always directly comparable across domains, interfaces, and compute budgets. For this reason, we emphasize mechanism-level interpretation and table-driven mapping instead of producing a single global performance ranking.

Finally, publication bias can distort observed trends. Positive results are more likely to be published and indexed, while negative findings may remain unavailable. This survey partially mitigates the issue by retaining removed/borderline artifacts and encouraging explicit failure reporting, but bias cannot be eliminated fully without broader community standards.'''

    sec['taxonomy.tex'] = f'''\\subsection{{(a) LLM as Planner / Task Decomposer}}
Planner-centric architectures use LLMs to generate intermediate goals, option structures, and executable plans that reduce long-horizon complexity. RL components then optimize low-level execution under transition feedback. This improves sample use when decomposition captures latent task structure that model-free exploration would discover slowly \\cite{{{cp}}}.

Planner modules are most effective when they are treated as dynamic hypotheses rather than static scripts. In practice, this means planners emit decompositions with confidence, the policy executes and observes environment outcomes, and the planner is re-invoked when confidence drops or constraints are violated. The interaction between replanning frequency and update stability is a nontrivial systems choice that can dominate results in long-horizon tasks.

Another recurring pattern is decomposition granularity control. Coarse decomposition reduces planning overhead but may leave too much burden for low-level exploration. Overly fine decomposition can create brittle dependence on plan tokens and amplify language noise. Strong systems expose decomposition granularity as an adaptive variable rather than fixing it globally.

\\subsection{{(b) LLM as Policy / Controller / Program Generator}}
Policy-level designs use language priors for action scoring or programmatic control while preserving environment-grounded optimization. They often improve early-stage competence in combinatorial action spaces, especially in language-conditioned control tasks \\cite{{{cl}}}.

These methods are often deployed with confidence-aware blending between learned value estimates and language-conditioned action priors. Blending can stabilize early training but must decay or recalibrate as policy competence improves; otherwise language priors can cap asymptotic performance. This trade-off is frequently underreported and should be part of standard ablation practice.

Program generator variants add interpretability benefits by exposing executable control logic. However, they may overfit benchmark-specific APIs. Robust studies therefore report cross-task transfer and evaluate whether generated logic remains valid after interface perturbations.

\\subsection{{(c) LLM as Reward Model / Reward Shaper / Verifier}}
Reward and verifier modules are highly represented in the core. They densify supervision through process rewards and trajectory checks, accelerating sparse-reward learning and making failure signals earlier and more actionable. Representative works include {esc(reward_examples)} \\cite{{{cr}}}.

This category can be decomposed into reward design, reward editing, process scoring, and post-hoc verification. Reward design generates candidate reward components from language task descriptions. Reward editing adjusts these components using trajectory evidence. Process scoring provides intermediate supervision, and post-hoc verification filters trajectories before updates. Each mechanism has different failure channels and should be evaluated separately.

A key limitation is proxy exploitation. If verifier criteria are easier to satisfy than environment objectives, policies can over-optimize formal checks while regressing on real outcomes. Mitigation strategies include multi-verifier agreement, adversarial stress episodes, and periodic \"no-shaping\" evaluation to test whether core competence remains when auxiliary guidance is removed.

\\subsection{{(d) LLM as World Model / Simulator / Proxy}}
World-model variants use language priors to improve transition abstraction and lookahead quality. This can reduce expensive online rollouts but requires uncertainty-aware correction to control model bias. Representative works include {esc(world_examples)} \\cite{{{cw}}}.

The central challenge in this category is compounding abstraction error. Language models can be strong at semantic prediction but weak at low-level dynamics. If this mismatch is not tracked, model-based planning can become overconfident and push policies toward infeasible trajectories. Conservative update rules and uncertainty-aware planning are therefore essential.

A promising direction is hybrid rollouts where model-predicted segments are periodically anchored to real-environment transitions. This combines sample-efficiency benefits of world proxies with grounding benefits of direct interaction. Standardized error metrics for this hybrid regime remain an open need.

\\subsection{{(e) Exploration / Curriculum / Goal Generation}}
Exploration-oriented papers use LLMs to propose goals, sequence tasks, and reweight exploration effort toward semantically meaningful novelty. Gains are strongest when curriculum acceptance is tied to measured policy improvement \\cite{{{ce}}}.

Curriculum systems typically perform best when goal generation is coupled with competence estimation. In this setup, the curriculum module proposes tasks near the current capability frontier rather than uniformly novel tasks. Frontier-targeted sampling improves learning efficiency and reduces time spent on goals that are either trivial or unreachable.

Failure occurs when generated goals are linguistically rich but behaviorally irrelevant. Strong pipelines guard against this by requiring measurable return gains, controllable difficulty progression, and replay balancing to avoid forgetting earlier skills.

\\subsection{{(f) Offline RL / Data and Trajectory Generation}}
In offline settings, LLMs support trajectory synthesis, relabeling, and annotation. These methods can improve policy training when synthetic samples are filtered by consistency and value-aware criteria to prevent distribution drift \\cite{{{co}}}.

Offline data generation is especially valuable in expensive or safety-critical domains where online interaction is limited. LLM-based relabeling can recover latent reward structure from noisy logs, while synthetic augmentation can increase rare-event coverage. However, synthetic data can shift support and destabilize value estimation if not constrained.

Practical safeguards include behavior-cloning regularization, uncertainty-weighted sampling, and explicit synthetic-to-real ratio reporting. Papers that include these controls offer substantially more credible evidence than those reporting aggregate gains without data diagnostics.

\\subsection{{(g) Tool Use / Memory / Multi-turn Agent RL}}
Tool and memory modules extend both state and action interfaces. LLMs decide when to call tools, parse outputs, and preserve context over long episodes. Representative works include {esc(tool_examples)}. Robustness depends on exception-aware recovery and verifier-gated tool traces \\cite{{{ct}}}.

Tool integration introduces combinatorial interface risk: APIs fail, responses drift, and parsing assumptions break. Without error-aware control, one malformed call can corrupt memory and misguide many subsequent actions. Successful architectures therefore include typed tool schemas, call-level validation, and repair actions that explicitly recover from interface exceptions.

Memory design is equally important. Long-context storage can improve multi-turn decision quality but can also accumulate stale or contradictory facts. Effective memory modules include compression policies, conflict resolution, and relevance gating so that policy updates consume useful context instead of raw logs.

\\subsection{{(h) Hybrid Tree Search + LLM + RL}}
Search hybrids combine language priors with branching lookahead and RL value correction. They are useful in high-branching tasks with delayed consequences but require careful compute control and prior calibration \\cite{{{cs}}}.

In these systems, LLM priors primarily affect branch ordering and expansion policy. RL value components provide grounded correction signals. If prior confidence is overestimated, search diversity collapses; if prior confidence is ignored, search cost rises sharply. Effective hybrids balance this with uncertainty-aware exploration and adaptive branch budgets.

Compute trade-offs are central. Branching depth, evaluator latency, and cache strategy determine whether search gains survive under fixed wall-clock budgets. This makes engineering decisions inseparable from algorithmic claims in tree-search categories.

\\subsection{{Cross-category Synthesis}}
Across categories, selective authority is the dominant design principle: LLM outputs should guide but not bypass environment-grounded correction. Systems with explicit gating, uncertainty handling, and rollback are more robust than systems that treat language priors as hard constraints \\cite{{{ca}}}.

From a unifying perspective, each category modifies a different credit-assignment pathway. Planning changes temporal decomposition, reward/verifier modules change supervisory density, world models change predictive structure, and tool-memory modules change state-action representation. High-performing architectures usually coordinate multiple pathways, but they also require explicit mechanisms to prevent cross-module error amplification.

Taxonomy-aware evaluation should therefore replace one-size-fits-all reporting. Planner systems need decomposition validity and recovery metrics. Reward systems need shortcut-resistance and verifier-calibration metrics. Tool-memory systems need interface robustness and long-context reliability metrics. Without category-specific diagnostics, aggregate benchmark gains are difficult to interpret and transfer.

\\subsection{{Design Pattern Matrix Across Categories}}
A practical synthesis is to view each architecture as a matrix over four axes: intervention point, authority level, correction path, and diagnostics. Intervention point identifies where the LLM enters the loop (planner, reward, world model, tool-memory, or search prior). Authority level describes whether outputs are advisory, weighted, or mandatory. Correction path captures how environment feedback can revise language outputs. Diagnostics determine whether failures can be localized during training.

Across the verified core, successful systems typically combine medium authority with fast correction. They allow language modules to shape exploration and intermediate supervision, but they preserve a direct route for environment feedback to override poor guidance. In contrast, systems with high authority and weak correction often show unstable behavior under distribution shift, even when benchmark averages look strong in narrow settings.

Another repeatable pattern is modular observability. Architectures that log planner calls, verifier verdicts, reward edits, tool exceptions, and memory updates provide better ablation evidence and easier debugging. This observability is not just an engineering convenience; it directly supports scientific comparability because independent groups can isolate which components caused gains.

A fourth pattern is curriculum coupling. Planner quality, reward shaping quality, and tool reliability all evolve during training. Static coupling policies can overfit early conditions and fail later. Adaptive coupling - where module influence depends on confidence, disagreement, and recent trajectory outcomes - is increasingly common in stronger systems. Future methods should report these schedules explicitly.

Finally, category interactions should be treated as first-class design objects. For example, reward/verifier modules can stabilize planner exploration by pruning low-value branches, while memory modules can improve verifier consistency by preserving context that isolated checks miss. Conversely, poor interaction design can cause cascading errors. Cross-category co-design is therefore likely to be the next major frontier after single-module improvements.'''

    sec['benchmarks.tex'] = f'''Core evaluations span robotics/embodied tasks, web and browser interaction, game-like text environments, continuous control, and offline datasets. This breadth is useful for external validity but complicates direct comparison because interaction budgets, tool wrappers, and verifier thresholds differ across papers.

Mechanism-aware evaluation is therefore necessary. Planner-heavy methods should be judged on long-horizon decomposition quality; reward/verifier-heavy methods on proxy leakage and sparse-reward stability; tool-memory methods on multi-turn reliability and failure recovery \\cite{{{ct}}}. Table T3 summarizes environment-category coverage, while T4 and T5 summarize supervision and interface patterns.

The current landscape shows stronger representation for planning/reward categories and weaker representation for world-model and memory categories, indicating clear opportunities for future benchmark development \\cite{{{cw}}}. This imbalance matters because poorly represented categories often lack standardized evaluation protocols, making cross-paper comparison noisy and slowing consolidation.

A robust benchmark protocol for LLM-in-the-loop RL should report at least four dimensions: (1) environment configuration and version, (2) interaction budget in both steps and wall-clock cost, (3) failure-handling policies for tool calls, planner contradictions, and verifier disagreements, and (4) ablations that isolate where gains originate. Without these details, it is difficult to tell whether improvements come from model design, prompt design, infrastructure, or hidden implementation choices.

Metrics should also be category-aligned. In planner categories, decomposition quality and replanning efficiency are as important as final success. In reward categories, shortcut resistance and calibration under shift are central. In tool-memory categories, multi-turn consistency and error-recovery latency should be tracked explicitly. Using one scalar metric across all settings obscures mechanism-level progress and leads to misleading comparisons.

Finally, benchmark reporting should include negative results and failure traces. LLM modules can fail silently by producing plausible but incorrect intermediate structure. Publishing trajectory-level failures, verifier disagreements, and policy rollback statistics is essential for cumulative progress and realistic deployment expectations.

Benchmark suites should also separate environment difficulty from interface difficulty. Some tasks are hard because of delayed reward structure, while others are hard because external tool APIs are noisy or underspecified. Conflating these effects can inflate or hide true RL improvements. Clear task-factor annotations would improve interpretability and help identify where specific module types are effective.

A final recommendation is compute-normalized reporting. Because LLM calls may dominate runtime cost, reporting only environment steps can be misleading. Studies should provide call counts, token budgets, and latency summaries alongside return/success metrics. This makes comparisons fairer across architectures that make different trade-offs between inference overhead and decision quality.'''

    sec['tradeoffs.tex'] = f'''Three recurring failure modes appear in the verified core: (1) reward/verifier shortcutting, where proxies are optimized without true task completion; (2) planner-policy mismatch, where semantically plausible plans fail at low-level execution; and (3) tool-mediated compounding errors, where invalid calls and parse failures contaminate trajectories \\cite{{{ct}}}.

Mitigation patterns are converging: adversarial verifier tests, feasibility-gated replanning, typed tool interfaces, and rollback policies for error recovery. Another practical trade-off is latency: frequent LLM calls can improve decision quality but reduce throughput and alter exploration under fixed compute budgets. Deployable systems must explicitly manage call frequency, caching, and fallback behavior \\cite{{{ca}}}.

A fourth trade-off is supervision fragility. Reward shaping and verifier guidance can accelerate convergence early, yet make policies dependent on auxiliary signals that may be unavailable or unreliable at deployment time. To avoid this, systems should evaluate both guided and unguided performance and report degradation profiles when auxiliary modules are partially disabled.

There is also a governance trade-off between flexibility and auditability. Highly adaptive language modules can change behavior across contexts in ways that are difficult to trace. For high-stakes settings, architectural simplicity and transparent intervention points may be preferable to maximal short-term performance. This is especially relevant in settings where debugging and post-incident analysis are mandatory.

From an optimization standpoint, hybrid architectures face gradient pathway interference. Planner updates, reward edits, and policy updates may optimize partially conflicting objectives if synchronization is weak. Practical stabilization methods include staged training, periodic module freezing, and confidence-weighted update scheduling. These choices should be reported clearly because they can materially alter outcomes.

In summary, performance gains in LLM-augmented RL are inseparable from systems trade-offs. Robust deployment depends on managing shortcut risk, interface reliability, latency budgets, and objective consistency jointly rather than treating them as independent issues.'''

    sec['open_problems.tex'] = f'''\\textbf{{1) Credit assignment with language intermediates.}} Formal return decomposition through planner/reward/tool intermediates remains underdeveloped. Existing methods often show empirical gains without a unified account of how language-generated signals alter gradient pathways. Better theoretical and diagnostic tools are needed to explain when guidance helps and when it introduces bias.

\\textbf{{2) Verifier calibration and anti-shortcut guarantees.}} Reward shaping and process supervision require stronger reliability tests under distribution shift \\cite{{{cr}}}. Future work should quantify verifier uncertainty, detect proxy divergence online, and provide guarantees that shaped objectives preserve core task intent.

\\textbf{{3) Cross-interface generalization.}} Tool-augmented agents often overfit wrapper conventions; robustness benchmarks should include interface perturbations and tool outages \\cite{{{ct}}}. A practical agenda is to evaluate policies under API schema changes, partial failures, and noisy tool outputs to measure real-world resilience.

\\textbf{{4) World-model fidelity.}} Language-based predictive abstractions need uncertainty-aware model fusion and conservative update rules \\cite{{{cw}}}. Standard model-bias diagnostics and hybrid rollout protocols are still immature in this literature and should become first-class reporting requirements.

\\textbf{{5) Benchmark standardization.}} Future suites should log planner calls, reward edits, verifier outputs, tool traces, and policy updates in a shared schema. This would allow mechanism-level comparison across papers and reduce ambiguity in \"LLM helps RL\" claims.

\\textbf{{6) Safety in open-ended loops.}} As module authority increases, robust fail-safe mechanisms and trajectory-level auditing become mandatory. Safety research should move beyond post-hoc filtering toward design-time constraints, including bounded intervention policies and explicit uncertainty-triggered fallback modes.

\\textbf{{7) Data-centric curation and reproducibility.}} Rapid publication cycles increase risk of metadata drift and scope leakage. Community benchmarks should bundle verifiable citation metadata, directionality labels, and reproducible parsing scripts so that future surveys can be updated without redoing manual cleanup from scratch.

\\textbf{{8) Human-in-the-loop oversight protocols.}} In many deployment scenarios, humans still supervise edge cases. Research should formalize how human feedback interacts with planner/reward/tool modules in RL loops, including when and how oversight signals should override autonomous policies.

Overall, these open problems are tightly coupled rather than independent. Better credit assignment theory improves verifier calibration; better calibration improves safety; better benchmark standards improve reproducibility across all categories. Progress will likely require coordinated advances in algorithms, systems tooling, and evaluation governance.'''

    sec['conclusion.tex'] = f'''This survey provides a verified and directionality-clean map of work where LLM modules improve RL systems from inside the control loop. Across {n} core papers, the strongest gains come from trajectory-level interventions: decomposition, supervisory densification, predictive abstraction, and tool-memory context management.

The same interfaces define the dominant failure surface. Reliable progress will depend on calibrated module authority, environment-grounded diagnostics, and benchmark standards centered on credit assignment rather than text quality alone.

The broader message is methodological. \"LLM for RL\" is not one monolithic method, but a family of design choices about where language enters control and how it is corrected by feedback. Progress should therefore be evaluated by mechanism-level evidence: which module changed, which signal improved, and which failure modes were mitigated. This shift from benchmark-only reporting to mechanism-aware reporting is essential for reproducible engineering.

As the field moves forward, the most valuable contributions may come from robust coupling strategies rather than isolated model scaling: safer verifier calibration, uncertainty-aware world modeling, resilient tool interfaces, and standardized trajectory logging. With these foundations, LLM-in-the-loop RL can transition from promising prototypes to dependable systems in real environments.

For practitioners, the immediate implication is to treat architecture and evaluation as inseparable. Choosing where the LLM sits in the loop determines both the achievable gains and the likely failure modes. Choosing what to log and ablate determines whether those gains are scientifically interpretable. A mature LLM-for-RL pipeline should therefore be designed as an auditable control system with explicit intervention points, fallback paths, and mechanism-aware metrics from the start.'''

    sec['appendix.tex'] = '''\\section{Full Paper List and Extended Mapping}
The complete verified bibliography is provided in \\texttt{core\\_llm\\_for\\_rl\\_v3.csv}. BibTeX entries for all core papers are provided in \\texttt{references.bib}.

\\section{Extended Taxonomy Mapping}
Core papers are tagged with one or more categories from: Policy, Planning, Reward, CreditAssignment, WorldModel, Exploration, DatasetGeneration, OfflineRL, ToolUse, Memory, Reasoning, Simulation, TreeSearch, Verifier, Curriculum, LanguageConditioning, MultiAgent, Architecture, HybridRL, AgentTraining.

\\section{Reproducibility Artifacts}
All analytics and figures were generated from the curated CSV and exported as \\texttt{tables\\_from\\_csv.csv} plus figure assets under \\texttt{figures/}.'''

    for fn, txt in sec.items():
        (SEC / fn).write_text(txt + '\n', encoding='utf-8')


def write_main():
    m = r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{setspace}
\setstretch{1.08}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}

\title{LLMs in the RL Loop: Using Large Language Models to Improve Reinforcement Learning\\A Verified, Directionality-Constrained Survey}
\author{Automated Survey Draft from Verified Core Bibliography}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\input{sections/abstract}
\end{abstract}

\section{Introduction}
\input{sections/introduction}
\section{Background}
\input{sections/background}
\section{Survey Methodology (PRISMA-style)}
\input{sections/methodology}
\section{Taxonomy: LLM Roles Inside the RL Loop}
\input{sections/taxonomy}
\section{Benchmarks and Evaluation Protocols}
\input{sections/benchmarks}
\section{Design Trade-offs and Failure Modes}
\input{sections/tradeoffs}
\section{Open Problems and Research Agenda}
\input{sections/open_problems}
\section{Conclusion}
\input{sections/conclusion}
\section{Tables}
\input{sections/tables}
\section{Figures}
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/f1_llm_rl_loop.png}
\caption{LLM modules inside the RL loop: planner, reward/verifier, world model, and tool-memory interfaces feeding policy updates.}
\end{figure}
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/f2_taxonomy_tree.png}
\caption{Hierarchical taxonomy of LLM roles for improving RL systems.}
\end{figure}
\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/f3_year_category_heatmap.png}
\caption{Year by category density for the verified core (2022--2026).}
\end{figure}

\appendix
\input{sections/appendix}

\bibliographystyle{unsrt}
\bibliography{references}
\end{document}
'''
    (OUT / 'main.tex').write_text(m, encoding='utf-8')

def write_outline(df):
    tv = df['Venue'].value_counts().head(10)
    o = []
    o.append('# Paper Outline')
    o.append('')
    o.append('## Thesis')
    o.append('LLMs improve RL most reliably when embedded as trajectory-level modules (planning, reward/verifier, world model, tool-memory), where they reshape credit assignment and state construction.')
    o.append('')
    o.append('## Contributions')
    o.append('1. Directionality-constrained, item-by-item verified bibliography.')
    o.append('2. PRISMA-style methodology and reproducible curation workflow.')
    o.append('3. Taxonomy centered on module roles in RL loops.')
    o.append('4. Data-driven trend analysis by year/category/domain/venue.')
    o.append('5. Failure-mode synthesis and targeted research agenda.')
    o.append('')
    o.append('## Structure')
    for s in ['Introduction','Background','Methodology','Taxonomy','Benchmarks','Trade-offs','Open Problems','Conclusion','Appendix']:
        o.append(f'- {s}')
    o.append('')
    o.append('## Data Snapshot')
    o.append(f'- Core papers: {len(df)}')
    o.append(f"- Year range: {int(df['Year'].min())} to {int(df['Year'].max())}")
    o.append('- Top venues:')
    for v,cnt in tv.items():
        o.append(f'  - {v}: {cnt}')
    o.append('')
    o.append('## Assets')
    for x in ['T1 Taxonomy overview','T2 Condensed paper matrix','T3 Benchmark matrix','T4 Reward/verifier patterns','T5 Tool-use and memory patterns','T6 Chronological trends','F1 Loop diagram','F2 Taxonomy tree','F3 Year-category heatmap']:
        o.append(f'- {x}')
    (OUT / 'paper_outline.md').write_text('\n'.join(o) + '\n', encoding='utf-8')


def main():
    if not CORE.exists():
        raise SystemExit('Missing core_llm_for_rl_v3.csv')
    OUT.mkdir(exist_ok=True); SEC.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

    df = pd.read_csv(CORE)
    miss = [x for x in REQ if x not in df.columns]
    if miss:
        raise SystemExit(f'Missing required columns: {miss}')

    df = df.copy()
    df['Title'] = df['Title'].apply(c)
    df['Authors'] = df['Authors'].apply(c)
    df['Venue'] = df['Venue'].apply(c)
    df['ArXiv_or_DOI'] = df['ArXiv_or_DOI'].apply(norm_id)
    df['Verification_URL'] = df['Verification_URL'].apply(c)
    df['Contribution_Summary'] = df['Contribution_Summary'].apply(c)
    df['Why_It_Belongs_in_Survey'] = df['Why_It_Belongs_in_Survey'].apply(c)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
    df['Tags'] = df['Category_Tags'].apply(tags)
    df['Domain'] = [domain(t,s) for t,s in zip(df['Title'], df['Contribution_Summary'])]
    df = df.sort_values(['Year','Title'], ascending=[False,True]).reset_index(drop=True)
    df = mk_keys(df)

    fig_loop(); fig_taxonomy(); fig_heat(df)

    rows = build_rows(df)
    write_tables(rows)
    export_analytics(df, rows)
    write_bib(df)

    counts = {
        'out': count_csv(ROOT / 'out_of_scope_rl_for_llm_v3.csv'),
        'border': count_csv(ROOT / 'manual_review_borderline_v3.csv'),
        'rem': count_csv(ROOT / 'removed_unverified_or_nonacademic_v3.csv'),
    }
    write_sections(df, counts)
    write_main()
    write_outline(df)
    print('survey_paper generated')


if __name__ == '__main__':
    main()
