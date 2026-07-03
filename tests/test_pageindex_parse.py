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
