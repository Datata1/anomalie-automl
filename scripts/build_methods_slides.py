"""Baut das Methoden-Deck als .pptx aus ``docs/methoden_praesentation.md``.

Parst die Marp-Markdown (Folien per ``---`` getrennt) und bettet die zuvor mit mermaid-CLI
gerenderten Diagramm-PNGs (``reports/diagrams/*.png``) ein. So entsteht eine echte .pptx mit
sichtbaren Diagrammen — ohne dass Marp mermaid nativ rendern muss.

Voraussetzung: die Diagramme wurden bereits gerendert, z. B.:
  npx @mermaid-js/mermaid-cli -i docs/methoden_praesentation.md \
      -o reports/diagrams/methoden_rendered.md -p /tmp/pptr.json -b white

Ausführen:  uv run python scripts/build_methods_slides.py
Ergebnis:   reports/methoden_praesentation.pptx
"""

from __future__ import annotations

import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "methoden_praesentation.md"
DIAGRAMS = ROOT / "reports" / "diagrams"
OUT = ROOT / "reports" / "methoden_praesentation.pptx"

SW, SH = Inches(13.333), Inches(7.5)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x33, 0x33, 0x33)


def _clean(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^>\s*", "", line)
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"`(.+?)`", r"\1", line)
    line = re.sub(r"\$(.+?)\$", r"\1", line)
    return line.strip()


def _diagram_pngs() -> list[Path]:
    pngs = list(DIAGRAMS.glob("*.png"))

    def key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    return sorted(pngs, key=key)


def parse_slides(text: str) -> list[dict]:
    body = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    chunks = re.split(r"\n---\n", body)
    slides = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        title_m = re.search(r"^#\s+(.+)$", chunk, flags=re.M)
        sub_m = re.search(r"^##\s+(.+)$", chunk, flags=re.M)
        has_diagram = "```mermaid" in chunk

        # Tabelle (Markdown) erkennen
        table = [
            [c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in chunk.splitlines()
            if ln.strip().startswith("|") and not re.match(r"^\|[-:\s|]+\|?$", ln.strip())
        ]

        # Bullets: ohne Codeblöcke, Überschriften, Tabellen
        no_code = re.sub(r"```.*?```", "", chunk, flags=re.S)
        bullets = []
        for ln in no_code.splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("|"):
                continue
            cleaned = _clean(s)
            if cleaned:
                bullets.append(cleaned)

        slides.append({
            "title": title_m.group(1).strip() if title_m else "",
            "subtitle": sub_m.group(1).strip() if sub_m else "",
            "bullets": bullets,
            "table": table,
            "has_diagram": has_diagram,
        })
    return slides


def _title_box(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), SW - Inches(1.0), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = text
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = ACCENT


def _bullets_box(slide, bullets, left, top, width, height, size=15):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = "• " + b
        run.font.size = Pt(size)
        run.font.color.rgb = GREY
        p.space_after = Pt(4)


def _table(slide, rows, left, top, width):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, Inches(0.4 * n_rows))
    tbl = shape.table
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            if para.runs:
                para.runs[0].font.size = Pt(11)
                para.runs[0].font.bold = i == 0


def build():
    slides = parse_slides(MD.read_text(encoding="utf-8"))
    diagrams = _diagram_pngs()
    di = 0

    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH

    for idx, s in enumerate(slides):
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        if idx == 0:  # Titelfolie
            box = slide.shapes.add_textbox(Inches(0.8), Inches(2.6), SW - Inches(1.6), Inches(2.2))
            tf = box.text_frame
            tf.word_wrap = True
            r = tf.paragraphs[0].add_run()
            r.text = s["title"]
            r.font.size = Pt(38)
            r.font.bold = True
            r.font.color.rgb = ACCENT
            if s["subtitle"]:
                p = tf.add_paragraph()
                rr = p.add_run()
                rr.text = s["subtitle"]
                rr.font.size = Pt(20)
                rr.font.color.rgb = GREY
            continue

        _title_box(slide, s["title"])
        png = None
        if s["has_diagram"] and di < len(diagrams):
            png = diagrams[di]
            di += 1

        if png is not None:
            _bullets_box(slide, s["bullets"], Inches(0.5), Inches(1.3), Inches(12.3), Inches(2.2), size=14)
            slide.shapes.add_picture(str(png), Inches(2.0), Inches(3.6), height=Inches(3.5))
        elif s["table"]:
            _bullets_box(slide, s["bullets"], Inches(0.5), Inches(1.3), Inches(12.3), Inches(1.2), size=14)
            _table(slide, s["table"], Inches(0.5), Inches(2.6), Inches(12.3))
        else:
            _bullets_box(slide, s["bullets"], Inches(0.6), Inches(1.4), SW - Inches(1.2), Inches(5.6), size=18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Gespeichert: {OUT}  ({len(prs.slides._sldIdLst)} Folien, {di} Diagramme eingebettet)")


if __name__ == "__main__":
    build()
