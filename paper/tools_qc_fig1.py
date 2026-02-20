import re
import sys
from itertools import combinations

import pdfplumber


EPSILON = 2.0


def intersection_area(a, b):
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["top"], b["top"])
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["bottom"], b["bottom"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def find_fig1_pages(pdf):
    pages = []
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if re.search(r"\bFig\.?\s*1\b|\bFigure\s*1\b", text):
            pages.append(idx)
    if pages:
        return pages

    fallback = []
    phrase = "LLM modules embedded in the reinforcement learning loop"
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if phrase.lower() in text.lower():
            fallback.append(idx)
    return fallback


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sn-article.pdf"
    with pdfplumber.open(pdf_path) as pdf:
        fig_pages = find_fig1_pages(pdf)
        if not fig_pages:
            print("QC ERROR: Could not find page containing Fig. 1 caption/text.")
            sys.exit(2)

        total_overlaps = 0
        examples = []
        for pidx in fig_pages:
            page = pdf.pages[pidx]
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            overlaps = 0
            for w1, w2 in combinations(words, 2):
                area = intersection_area(w1, w2)
                if area > EPSILON:
                    overlaps += 1
                    if len(examples) < 8:
                        examples.append((pidx + 1, w1["text"], w2["text"], round(area, 3)))
            print(f"Fig.1 page {pidx + 1}: overlap_count={overlaps}")
            total_overlaps += overlaps

        print(f"TOTAL_OVERLAP_COUNT={total_overlaps}")
        if total_overlaps > 0:
            print("Example overlapping pairs (page, word_a, word_b, area):")
            for ex in examples:
                print(ex)
            sys.exit(1)


if __name__ == "__main__":
    main()
