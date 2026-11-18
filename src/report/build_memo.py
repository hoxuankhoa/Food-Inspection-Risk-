"""Step 8 - render the policy memo (markdown) to Word with key figures appended.

Input:  reports/policy_memo/policy_memo.md (hand-written, numbers from the pipeline)
Output: reports/policy_memo/policy_memo.docx
"""
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

from src import config

FIGURES = [
    ("capture_curve.png", "Figure 1. Share of failed inspections captured vs. capacity, 2019."),
    ("reliability_plot.png", "Figure 2. Reliability plot of calibrated and uncalibrated scores, 2019."),
    ("fairness_region.png", "Figure 3. Selection and coverage by region at 10% capacity."),
]


def add_runs(paragraph, text: str) -> None:
    """Minimal inline markdown: **bold**, *italic*, `code`."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def add_table(doc: Document, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [r for r in cells[2:]]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        table.rows[0].cells[i].text = ""
        add_runs(table.rows[0].cells[i].paragraphs[0], f"**{h}**")
    for r in body:
        row = table.add_row().cells
        for i, v in enumerate(r):
            add_runs(row[i].paragraphs[0], v)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()


def main() -> None:
    lines = (config.MEMO_DIR / "policy_memo.md").read_text(encoding="utf-8").splitlines()
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.9)
        section.top_margin = section.bottom_margin = Inches(0.8)

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        if line.startswith("# "):
            h = doc.add_heading(line[2:], level=0)
            for run in h.runs:
                run.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif re.match(r"^\d+\. ", line):
            add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line))
        elif line.startswith("* "):
            add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
        elif line.strip():
            add_runs(doc.add_paragraph(), line)
        i += 1

    doc.add_page_break()
    doc.add_heading("Appendix: key figures", level=1)
    for name, caption in FIGURES:
        doc.add_picture(str(config.FIGURES / name), width=Inches(6.3 if "fairness" in name else 5.2))
        cap = doc.add_paragraph()
        add_runs(cap, f"*{caption}*")

    out = config.MEMO_DIR / "policy_memo.docx"
    doc.save(out)
    print(f"Memo -> {out}")


if __name__ == "__main__":
    main()
