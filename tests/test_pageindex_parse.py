import pytest

from hydra.pageindex.parse import parse_pdf

fpdf = pytest.importorskip("fpdf")  # dev dependency; skip cleanly if absent


def _make_pdf(path):
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "INTRODUCTION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "This document describes the annual results in detail.")
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "MARGINS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "Operating margin was 21.5 percent in 2023.")
    pdf.output(str(path))


def test_parse_pdf_recovers_sections_and_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)

    sections = parse_pdf(str(pdf_path))
    titles = [s.title for s in sections]
    assert "INTRODUCTION" in titles
    assert "MARGINS" in titles

    margins = next(s for s in sections if s.title == "MARGINS")
    assert margins.page == 1
    assert "21.5" in margins.content


def _make_layout_pdf(path):
    """3 pages, running header on every page, mixed-case headings detectable only
    by font size (no ALL-CAPS / numbering signal)."""
    pdf = fpdf.FPDF()
    bodies = [
        ("Liquidity and Capital", "Cash totaled 480 million at year end."),
        ("Results Overview", "Revenue grew 18 percent year over year."),
        ("Risk Summary", "Supplier concentration remains the top risk."),
    ]
    for title, body in bodies:
        pdf.add_page()
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6, "Contoso Corp Annual Report 2023", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=15)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, body)
    pdf.output(str(path))


def test_font_size_headings_and_running_header_filter(tmp_path):
    pdf_path = tmp_path / "layout.pdf"
    _make_layout_pdf(pdf_path)

    sections = parse_pdf(str(pdf_path))
    titles = [s.title for s in sections]
    # Mixed-case headings recovered via font size alone.
    assert "Liquidity and Capital" in titles
    assert "Risk Summary" in titles
    # The repeated running header never becomes a section or content.
    assert "Contoso Corp Annual Report 2023" not in titles
    assert all("Contoso Corp Annual Report" not in s.content for s in sections)
    # Page numbers survive.
    assert next(s for s in sections if s.title == "Risk Summary").page == 3
