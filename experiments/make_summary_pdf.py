"""
Generates the required one-page concept summary PDF.
Run: python experiments/make_summary_pdf.py
Output: memory-frontier/One_Page_Concept_Summary.pdf
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "One_Page_Concept_Summary.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body9", parent=styles["Normal"], fontSize=9.0, leading=11.8, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=13.5, leading=15.5, spaceAfter=5))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=9.8, leading=12, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#3a3a3a")))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=7.3, leading=9.4, textColor=colors.HexColor("#555555")))

doc = SimpleDocTemplate(
    OUT_PATH, pagesize=letter,
    topMargin=0.42*inch, bottomMargin=0.42*inch, leftMargin=0.6*inch, rightMargin=0.6*inch,
    title="Memory Frontier — One-Page Concept Summary"
)

story = []

story.append(Paragraph("Fixed-Size Synaptic Memory vs. the Growing KV Cache", styles["H1"]))
story.append(Paragraph(
    "Concept: Associative Memory and Fast Weights &nbsp;|&nbsp; DataForge 2026, Pathway Track", styles["Small"]))
story.append(Spacer(1, 4))

story.append(Paragraph("The design pressure", styles["H2"]))
story.append(Paragraph(
    "A standard Transformer answers a question about something said earlier by keeping every past token's key and "
    "value vector in a KV cache and comparing the current query against all of them. This is exact and simple, but "
    "the cache grows linearly with context length: doubling the conversation doubles the memory. Fast-weight and "
    "linear-attention research reframes this differently: instead of storing tokens, compress the past into a "
    "fixed-size state and update that state as an online learning rule (Sun et al., 2024, arXiv:2407.04620). The "
    "central open question this framing raises is not whether a fixed state can work — it clearly can, at some "
    "scale — but how its accuracy degrades as more information is written into a container that never grows.",
    styles["Body9"]))

story.append(Paragraph("What changes technically, and the trade-off", styles["H2"]))
story.append(Paragraph(
    "This project implements the simplest possible version of that idea: a dense Hebbian fast-weight matrix "
    "W (dim &times; dim), updated as W_t = &lambda;&middot;W_(t-1) + &eta;&middot;(v k<super>T</super>) and read as "
    "W&middot;query. Unlike the KV cache, W's size never changes with sequence length: it is O(1) in space for a "
    "fixed dimension, versus the KV cache's O(N). The cost is interference &mdash; new outer-product writes are "
    "superimposed on old ones in the same matrix, so old associations can be overwritten or blurred rather than "
    "evicted cleanly. On a controlled associative-recall benchmark built for this project (random key/value "
    "vectors, no semantic redundancy for either system to exploit), synaptic-memory retrieval accuracy on a fixed "
    "set of 20 tracked facts falls from 100% at 50 total facts to roughly 5&ndash;15% by 1,000&ndash;10,000 facts, "
    "while the KV cache stays at or near 100% throughout &mdash; at the cost of its memory growing roughly 300&times; "
    "over the same range. These are this project's own measurements (seed-fixed, reproducible via "
    "experiments/benchmark.py), not an independent evaluation.",
    styles["Body9"]))

story.append(Paragraph("Where this sits among recent systems", styles["H2"]))

cellstyle = ParagraphStyle(name="cell", parent=styles["Normal"], fontSize=7.1, leading=8.8)
def C(text):
    return Paragraph(text, cellstyle)

data = [
    ["System", "State w.r.t. length", "Handles interference via", "Evidence type"],
    [C("KV cache / full attention"), C("Grows, O(N)"), C("N/A (exact storage)"), C("Well-established baseline")],
    [C("Linear attention w/ fixed state (Arora et al., 2024, arXiv:2402.18668)"), C("Fixed, O(1)"), C("Better feature maps / hybrid recall-throughput tuning"), C("Benchmarked (public)")],
    [C("Dragon Hatchling (BDH) (Kosowski et al., 2025, arXiv:2509.26507)"), C("Fixed, O(1) (neuron-graph state)"), C("Sparse, non-negative, scale-free connectivity"), C("Benchmarked; reported long-context results")],
    [C("This prototype"), C("Fixed, O(1)"), C("None (dense, unconstrained)"), C("Own controlled benchmark")],
]
tbl = Table(data, colWidths=[1.35*inch, 0.95*inch, 1.75*inch, 1.35*inch])
tbl.setStyle(TableStyle([
    ("FONTSIZE", (0,0), (-1,-1), 7.1),
    ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee")),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ("LEFTPADDING", (0,0), (-1,-1), 3),
]))
story.append(tbl)
story.append(Spacer(1, 3))

story.append(Paragraph(
    "The comparison is not that one system is strictly better: linear-attention variants trade some recall for "
    "throughput and are competitive when tuned (Arora et al., 2024); BDH's sparsity is specifically motivated by "
    "reducing the interference this project measures in its plainest, unconstrained form (Kosowski et al., 2025). "
    "A companion line of recent work studies the same forgetting problem directly and proposes an exact auxiliary "
    "memory to fix it (Fang et al., 2025, arXiv:2607.02303) &mdash; this project's degradation curve is best read as "
    "an illustration of the problem that motivates work like theirs, not a critique of any specific system.",
    styles["Body9"]))

story.append(Paragraph("Role of BDH and BDH-CQ", styles["H2"]))
story.append(Paragraph(
    "BDH's relevance is direct: it reformulates attention as Hebbian synaptic memory updated as the model reads, "
    "which is the same update family this prototype isolates, minus BDH's sparsity and scale-free connectivity "
    "constraints (Kosowski et al., 2025). BDH-CQ's relevance is more specific: it stores task demonstrations in "
    "recurrent memory at inference time and reasons iteratively in a continuous latent workspace rather than a "
    "verbalized chain-of-thought, reaching 29.5% pass@2 on the public ARC-AGI-1 evaluation set at roughly "
    "$0.0007 per task &mdash; a result independently reproduced by a team from Bielik AI and NYU, which is a genuine "
    "independent evaluation, distinct from a benchmark run by the system's own developers. BDH-CQ's contribution "
    "to this project is conceptual (adaptation living in state, not weights) rather than a component this "
    "prototype reuses.",
    styles["Body9"]))

story.append(Paragraph("Most important limitation", styles["H2"]))
story.append(Paragraph(
    "This prototype's dense, unconstrained Hebbian update is a deliberately weak baseline: it has none of the "
    "structural features (sparsity, non-negativity, monosemantic synapses) that BDH's own paper connects to lower "
    "interference. Its degradation curve should be read as an upper bound on the problem fixed-size memory faces "
    "in the worst case, not as a measurement of how BDH itself would perform on the same task &mdash; that "
    "comparison remains untested here and is the natural next experiment.",
    styles["Body9"]))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "Sources: Kosowski et al. 2025 (arXiv:2509.26507); Pathway 2026, BDH-CQ (arXiv:2608.09888); "
    "Arora et al. 2024 (arXiv:2402.18668); Sun et al. 2024 (arXiv:2407.04620); Fang et al. 2025 (arXiv:2607.02303). "
    "Full citations and reproduction instructions in README.md.",
    styles["Small"]))

doc.build(story)
print(f"Saved: {OUT_PATH}")
