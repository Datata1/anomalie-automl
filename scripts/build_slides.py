"""Erzeugt die Projekt-PowerPoint (python-pptx) aus den Figuren in ``reports/``.

~17 Folien, Deutsch, 16:9. Bettet die vorhandenen PNGs ein und ergänzt Bullet-Punkte +
eine Ergebnistabelle aus ``reports/results.csv``.

Ausführen:  uv run python scripts/build_slides.py
Ergebnis:   reports/automl_ad_praesentation.pptx
"""

from __future__ import annotations

import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

REPORTS = Path(__file__).resolve().parent.parent / "reports"
OUT = REPORTS / "automl_ad_praesentation.pptx"

SW, SH = Inches(13.333), Inches(7.5)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x40, 0x40, 0x40)


def _img(name: str) -> Path | None:
    p = REPORTS / name
    return p if p.exists() else None


def _title_box(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), SW - Inches(1.0), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = ACCENT


def _bullets_box(slide, bullets, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = "• " + b
        run.font.size = Pt(16)
        run.font.color.rgb = GREY
        p.space_after = Pt(6)


def _add_images(slide, images, left, top, total_w, total_h):
    imgs = [p for p in (_img(n) for n in images) if p is not None]
    if not imgs:
        return
    if len(imgs) == 1:
        slide.shapes.add_picture(str(imgs[0]), left, top, width=total_w)
    else:
        gap = Inches(0.1)
        each_h = (total_h - gap * (len(imgs) - 1)) / len(imgs)
        y = top
        for p in imgs:
            slide.shapes.add_picture(str(p), left, y, height=each_h)
            y += each_h + gap


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.6), SW - Inches(1.6), Inches(2.0))
    tf = box.text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = ACCENT
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = subtitle
    r2.font.size = Pt(20)
    r2.font.color.rgb = GREY


def add_content_slide(prs, title, bullets=None, images=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title_box(slide, title)
    bullets = bullets or []
    images = images or []
    content_top, content_h = Inches(1.4), Inches(5.6)
    if bullets and images:
        _bullets_box(slide, bullets, Inches(0.5), content_top, Inches(5.0), content_h)
        _add_images(slide, images, Inches(5.8), content_top, Inches(7.0), content_h)
    elif images:
        _add_images(slide, images, Inches(1.5), content_top, Inches(10.3), content_h)
    else:
        _bullets_box(slide, bullets, Inches(0.7), content_top, SW - Inches(1.4), content_h)
    return slide


def add_table_slide(prs, title, note):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title_box(slide, title)

    rows_data = []
    csv_path = REPORTS / "results.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows_data.append(row)

    headers = ["strategy", "method", "roc_auc", "pr_auc", "f1"]
    n_rows = len(rows_data) + 1
    table = slide.shapes.add_table(
        n_rows, len(headers), Inches(0.6), Inches(1.4), Inches(8.5), Inches(5.0)
    ).table
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
    for i, row in enumerate(rows_data, start=1):
        for j, h in enumerate(headers):
            val = row[h]
            try:
                val = f"{float(val):.3f}"
            except (ValueError, TypeError):
                pass
            cell = table.cell(i, j)
            cell.text = str(val)
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(11)

    _bullets_box(slide, [note], Inches(9.4), Inches(1.6), Inches(3.4), Inches(4.0))


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH

    add_title_slide(
        prs,
        "AutoML für Anomaliedetection in der Prozessindustrie",
        "Tennessee Eastman Process (TEP) — Methoden, Strategien & Ergebnisse",
    )

    add_content_slide(prs, "Motivation: Prozessüberwachung → AD → AutoML", bullets=[
        "Prozessüberwachung erkennt Fehler/ineffiziente Zustände aus Sensordaten (Predictive Maintenance).",
        "Anomaliedetection (AD): lerne Normalverhalten, markiere Abweichungen — univariat, multivariat, zeitlich.",
        "AutoML automatisiert Modellwahl, Hyperparameter, Vorverarbeitung (CASH).",
        "Frage: Wie nutzt man AutoML für AD — besonders, wenn Labels fehlen?",
    ])

    add_content_slide(prs, "Datensatz & EDA: Tennessee Eastman Process", bullets=[
        "52 Features (41 Messungen xmeas, 11 Stellgrößen xmv), 20 Fehlertypen.",
        "Zeitreihe je Lauf; im Test setzt der Fehler erst ab Sample 161 ein (Onset).",
        "Starke Sensor-Korrelationen; Fehlertypen in 2D (PCA) teils separierbar.",
    ], images=["00_eda_correlation.png", "00_eda_pca.png"])

    add_content_slide(prs, "Aufgabe: 3 Settings × 4 AutoML-Strategien", bullets=[
        "Settings: unsupervised AD, semi-supervised, supervised Klassifikation.",
        "Strategien: (1) HPO, (2) fertige Frameworks, (3) Meta-Learning/Selektion, (4) Ensembling.",
        "Kernproblem: im unüberwachten Fall fehlen Labels zur Modellselektion/HPO.",
        "TEP-Labels dienen als Oracle-Obergrenze zum Vergleich.",
    ])

    add_content_slide(prs, "Evaluationsprotokoll", bullets=[
        "Split nach simulationRun (kein Leakage); Scaler nur auf Gutdaten.",
        "Onset-korrektes Label: faultNumber≠0 ≠ jede Zeile anomal (erst ab Sample 161).",
        "Metriken: ROC-AUC, PR-AUC, F1 (threshold-frei) + Detection-Delay, False-Alarm-Rate.",
        "Score-Konvention: höher = anomaler; Threshold via Kontaminations-Quantil.",
    ])

    add_content_slide(prs, "Baselines: unüberwachte Detektoren (Defaults)", bullets=[
        "PyOD-Detektoren nur auf Gutdaten trainiert; ROC-AUC 0.78–0.855.",
        "Score springt exakt am Fehler-Onset über den Threshold (rechts).",
    ], images=["summary_defaults_roc_auc.png", "01_baselines_timeseries_fault1.png"])

    add_content_slide(prs, "Strategie 1: HPO (Bayesian Optimization)", bullets=[
        "Optuna/TPE tunt Detektoren gegen ein test-disjunktes Val-Set.",
        "Gewinn vs. Default moderat (starke Bibliotheks-Defaults).",
        "Trial-Verteilung: viele Configs sind schlecht — HPO findet zuverlässig gute.",
    ], images=["02_hpo_default_vs_tuned.png", "02_hpo_ocsvm_trial_distribution.png"])

    add_content_slide(prs, "Multi-Fidelity: Random vs. BO vs. Hyperband", bullets=[
        "Budget = Trainings-Subsample; Hyperband prunt aussichtslose Trials früh.",
        "Vergleichbare Qualität, aber effizienter (mehr Configs pro Budget).",
    ], images=["08_multifidelity_best_value.png", "08_multifidelity_elapsed.png"])

    add_content_slide(prs, "Strategie 4: Ensembling", bullets=[
        "Score-Normalisierung + Aggregation (Mittelwert/Max), label-frei.",
        "Robust nahe am besten Einzeldetektor; Feature Bagging & greedy Selection ergänzt.",
        "Umgeht das Selektionsproblem teilweise.",
    ], images=["04_ensemble_vs_individual.png", "10_ensemble_selection.png"])

    add_content_slide(prs, "★ Strategie 3: Modellselektion OHNE Labels", bullets=[
        "Kernproblem: kein Gütesignal ohne Labels.",
        "Label-freie Konsens-/interne Selektion (SIREOS) ≈ Oracle (0.854 vs 0.855).",
        "Aussage: AutoML-für-AD wählt auf TEP fast optimal — ohne Labels.",
    ], images=["09_internal_vs_consensus_vs_oracle.png", "summary_label_free_vs_oracle.png"])

    add_content_slide(prs, "Strategie 2: Frameworks & Klassifikation vs. AD", bullets=[
        "FLAML (CASH) vs. Random Forest vs. AutoGluon (isoliert, py3.12).",
        "Mit vielen Labels schlägt Klassifikation die unüberwachte AD …",
        "… erkennt aber nur BEKANNTE Fehler; AD generalisiert auf neue.",
    ], images=["03_classification_vs_ad.png", "12_framework_comparison.png"])

    add_content_slide(prs, "Zeitliche AD: LSTM-Autoencoder", bullets=[
        "Trainiert auf Gutdaten-Sequenzen; Score = Rekonstruktionsfehler eines Fensters.",
        "ROC-AUC ~0.99 inkl. Drift-Fehler IDV 13 — Dynamik schlägt punktweise klar.",
    ], images=["05_lstm_ae_timeseries_fault13.png"])

    add_content_slide(prs, "Semi-supervised: DeepSAD", bullets=[
        "Deep SVDD (0 Labels) vs. DeepSAD (wenige gelabelte Anomalien).",
        "DeepSAD 0.985 vs. Deep SVDD 0.834 — wenige Labels bringen großen Sprung.",
    ], images=["06_deepsad_vs_deepsvdd.png"])

    add_content_slide(prs, "Per-Fault-Deep-Dive: Welche Methode fängt welchen Fehler?", bullets=[
        "Leichte Fehler (1/2/6/13): alle hoch. Schwere (3/9/15): alle niedrig.",
        "Komplementarität: Fehler 4 nur von ocsvm/pca/AE/som erkannt — Motivation für Ensembles.",
    ], images=["11_per_fault_heatmap.png"])

    add_table_slide(
        prs, "Gesamtergebnis: AutoML-Strategien im Vergleich",
        "Label-frei (select_internal) ≈ Oracle (select_oracle). HPO-Gewinn moderat; Ensemble robust.",
    )

    add_content_slide(prs, "Diskussion: Warum diese Ergebnisse? Limitationen", bullets=[
        "Absolute AD-Zahlen durch harte Fehler (3/9/15) gedeckelt — Durchschnittseffekt.",
        "Starke Defaults → HPO-Gewinn klein; Subsampling/moderate Budgets für Tempo.",
        "AutoGluon/auto-sklearn: unter Python 3.13 nicht installierbar (numpy 2.x/pandas 3.x);",
        "  AutoGluon daher in separatem py3.12-venv benchmarkt — löst aber nur das ÜBERWACHTE Problem.",
        "Interne Metriken korrelieren generell oft schwach; echtes MetaOD scheiterte (py3.13).",
    ])

    add_content_slide(prs, "Fazit & Future Work", bullets=[
        "Kernaussage: AutoML kann AD-Modelle auch ohne Labels nahezu optimal auswählen.",
        "Zeitmodelle (LSTM-AE) und wenige Labels (DeepSAD) bringen die größten Sprünge.",
        "Besser machbar: fault-aware Ensembles/Routing, Zeitfenster für alle Detektoren,",
        "  größere HPO-Budgets, bessere interne Metriken, mehr Seeds/Volldaten, echtes MetaOD.",
        "Sauberes, reproduzierbares Example-Projekt (uv, marimo, PyOD-Interface, CI, Tests).",
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Gespeichert: {OUT}  ({len(prs.slides._sldIdLst)} Folien)")


if __name__ == "__main__":
    build()
