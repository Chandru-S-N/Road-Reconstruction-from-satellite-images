"""PDF report contents come from the session."""

from pypdf import PdfReader

from core.pdf_report import build_pdf
from tests.conftest import run_scene


def test_pdf_report(tmp_path):
    _, pipe = run_scene("07_multiple_components", use_mask=False)
    ids = [c.id for c in pipe.state.candidates]
    pipe.decide_gap(ids[0], "approve")
    path = build_pdf(pipe, tmp_path / "r.pdf")
    reader = PdfReader(str(path))
    text = "\n".join(p.extract_text() for p in reader.pages)
    assert len(reader.pages) >= 5
    assert f"Page 1 of {len(reader.pages)}" in text
    for gid in ids:
        assert f"Gap {gid}" in text
    assert "Successfully reconstructed" in text and "Pending" in text
