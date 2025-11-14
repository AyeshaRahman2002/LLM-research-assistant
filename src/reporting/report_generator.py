# src/reporting/report_generator.py
import hashlib
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfdoc
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


# Wrap hashlib.md5 so the unexpected kwarg is ignored.
def _md5_compat(data=b"", usedforsecurity=False):
    return hashlib.md5(data)


from reportlab.lib import utils as rl_utils

rl_utils.md5 = _md5_compat

pdfdoc.md5 = _md5_compat

OUT_MD = Path("reports/final_report.md")
OUT_PDF = Path("reports/final_report.pdf")


def read_text(p: Path, default=""):
    return p.read_text() if p.exists() else default


def build_markdown():
    lines = ["# Final Research Assistant Report\n"]
    # Abstract (short auto-abstract from metrics)
    sm = Path("results/summary_metrics.json")
    if sm.exists():
        m = json.loads(sm.read_text())
        lines += [
            f"**Abstract:** Generated {Path('results/summaries.json').exists() and 'summaries' or '—'} with ROUGE-L `{m.get('rougeL','')}` and BERTScore-F1 `{m.get('bertscore_f1','')}`.\n"
        ]
    # Literature
    lit_md = read_text(
        Path("reports/literature_summary.md"), "# Literature\n_Not generated yet._\n"
    )
    lines += ["\n## Literature Review Summary\n", lit_md]
    # Data analysis
    data_md = read_text(
        Path("results/data_analysis_commentary.md"),
        "# Data Analysis\n_Not generated yet._\n",
    )
    lines += ["\n## Data Analysis Results\n", data_md]
    # LLM efficiency
    if Path("results/benchmark_table.csv").exists():
        lines += [
            "\n## LLM Efficiency Benchmark\nSee `results/benchmark_table.csv` and `plots/tradeoff_curves.png`.\n"
        ]
    lines += [
        "\n## Discussion & Future Work\n- Expand datasets\n- Per-model accuracy metrics\n- Add factuality checks for RAG\n"
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE wrote -> {OUT_MD}")


def build_pdf():
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4, title="Final Research Assistant Report"
    )
    story = []

    def add_h(txt):
        story.append(Paragraph(f"<b>{txt}</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))

    def add_p(txt):
        story.append(Paragraph(txt.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))

    # sections
    add_h("Abstract")
    if Path("results/summary_metrics.json").exists():
        m = json.loads(Path("results/summary_metrics.json").read_text())
        add_p(
            f"ROUGE-L: {m.get('rougeL','')} | BERTScore-F1: {m.get('bertscore_f1','')}"
        )
    else:
        add_p("—")

    add_h("Literature Review Summary")
    add_p(read_text(Path("reports/literature_summary.md"), "—"))

    add_h("Data Analysis Results")
    add_p(read_text(Path("results/data_analysis_commentary.md"), "—"))

    add_h("LLM Efficiency Benchmark")
    if Path("plots/tradeoff_curves.png").exists():
        story.append(Image("plots/tradeoff_curves.png", width=400, height=260))
        story.append(Spacer(1, 8))
    add_p("See results/benchmark_table.csv for the full table.")

    add_h("Discussion & Future Work")
    add_p("Expand datasets; attach per-model accuracy; add factuality checks.")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(f"DONE PDF -> {OUT_PDF}")


if __name__ == "__main__":
    build_markdown()
    build_pdf()
