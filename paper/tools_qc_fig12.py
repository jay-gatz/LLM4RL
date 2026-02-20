import re
import sys
from itertools import combinations

import pdfplumber


EPSILON = 2.0
REGION_PADDING = 18.0


def intersection_area(a, b):
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["top"], b["top"])
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["bottom"], b["bottom"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def find_pages(pdf, regex_list, fallback_phrases):
    pages = []
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if any(re.search(rx, text) for rx in regex_list):
            pages.append(idx)
    if pages:
        return pages

    fallback = []
    for idx, page in enumerate(pdf.pages):
        text = (page.extract_text() or "").lower()
        if any(phrase.lower() in text for phrase in fallback_phrases):
            fallback.append(idx)
    return fallback


def overlap_count(page):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    count = 0
    examples = []
    for w1, w2 in combinations(words, 2):
        area = intersection_area(w1, w2)
        if area > EPSILON:
            count += 1
            if len(examples) < 5:
                examples.append((w1["text"], w2["text"], round(area, 3)))
    return count, examples


def normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def bbox_union(boxes):
    x0 = min(b["x0"] for b in boxes)
    top = min(b["top"] for b in boxes)
    x1 = max(b["x1"] for b in boxes)
    bottom = max(b["bottom"] for b in boxes)
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def expand_bbox(bb, pad):
    return {"x0": bb["x0"] - pad, "top": bb["top"] - pad, "x1": bb["x1"] + pad, "bottom": bb["bottom"] + pad}


def intersects(bb, w):
    return not (w["x1"] < bb["x0"] or w["x0"] > bb["x1"] or w["bottom"] < bb["top"] or w["top"] > bb["bottom"])


def overlap_count_in_region(page, anchors):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    norm_to_words = {}
    for w in words:
        key = normalize_token(w.get("text", ""))
        if not key:
            continue
        norm_to_words.setdefault(key, []).append(w)

    anchor_boxes = []
    for a in anchors:
        key = normalize_token(a)
        for w in norm_to_words.get(key, []):
            anchor_boxes.append(w)

    # Need a few anchors to localize the figure region reliably.
    if len(anchor_boxes) < 3:
        return None, None

    region = expand_bbox(bbox_union(anchor_boxes), REGION_PADDING)
    in_region = [w for w in words if intersects(region, w)]

    count = 0
    examples = []
    for w1, w2 in combinations(in_region, 2):
        area = intersection_area(w1, w2)
        if area > EPSILON:
            count += 1
            if len(examples) < 5:
                examples.append((w1["text"], w2["text"], round(area, 3)))
    return count, examples


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sn-article.pdf"

    targets = {
        "Fig1": {
            "regex": [r"\bFig\.?\s*1\b", r"\bFigure\s*1\b"],
            "fallback": ["LLM modules embedded in the reinforcement learning loop"],
            "anchors": ["Environment", "RL", "Agent", "Planner", "Reward", "World", "Tool", "Memory"],
        },
        "Fig2": {
            "regex": [r"\bFig\.?\s*2\b", r"\bFigure\s*2\b"],
            "fallback": ["Role-based taxonomy of LLM-enhanced RL"],
            "anchors": ["Information", "Reward", "Decision", "Generator"],
        },
    }

    total = 0
    failed = False

    with pdfplumber.open(pdf_path) as pdf:
        for name, cfg in targets.items():
            pages = find_pages(pdf, cfg["regex"], cfg["fallback"])
            if not pages:
                print(f"{name}: page_not_found")
                failed = True
                continue

            local_total = 0
            local_examples = []
            for p in pages:
                # Prefer a localized region around the figure to avoid false positives from math typesetting elsewhere.
                cex = overlap_count_in_region(pdf.pages[p], cfg.get("anchors", []))
                if cex[0] is None:
                    c, ex = overlap_count(pdf.pages[p])
                else:
                    c, ex = cex
                local_total += c
                if ex and len(local_examples) < 5:
                    local_examples.extend(ex[: 5 - len(local_examples)])
                print(f"{name} page {p + 1}: overlap_count={c}")

            total += local_total
            if local_total > 0:
                failed = True
                print(f"{name} examples: {local_examples}")

    print(f"TOTAL_OVERLAP_COUNT={total}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
